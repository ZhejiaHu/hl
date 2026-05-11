//! Constrained fractional Kelly portfolio optimizer with joint multi-asset
//! combo optimization.
//!
//! ## Joint optimization algorithm
//! 1. **Per-asset Kelly fractions**: for each asset with sufficient confidence,
//!    compute f*_i = kelly_fraction × μ_i / (σ_i²+μ_i²), adjusted for TC.
//! 2. **Candidate scoring**: net_score_i = ELG_i − turnover_i × tc_i (in fraction units).
//! 3. **Subset selection** (cardinality constraint max_assets):
//!    - If n_candidates ≤ 10 and enumerate_subsets: exhaustive C(n,k) search for
//!      the subset maximising total net_score subject to TC budget.
//!    - Otherwise: greedy rank by net_score, accumulate until TC budget exhausted.
//! 4. **Portfolio constraint scaling**: if gross or net exposure of selected legs
//!    exceeds limits, uniformly scale all fractions down.
//! 5. **Leg construction**: entry/stop/TP from the per-asset `AssetDistribution`.
//!
//! ## DP extension
//! `optimize_dp` uses per-asset backward induction (41-point grid) to compute
//! multi-period optimal fractions and then applies the same portfolio constraints.
//! The distribution is held constant over the horizon (recalibrated live).

use std::collections::HashMap;

use uuid::Uuid;

use crate::{
    ensemble::EnsembleDistribution,
    risk::manager::PortfolioState,
    signals::generator::Direction,
};

use super::{
    AssetDistribution, ComboConfig, ComboOrder, OrderLeg,
    PortfolioConstraints, PortfolioOptimizer, TradeDecision,
};

// ─────────────────────────────────────────────────────────────────────────────

pub struct KellyOptimizer;

impl KellyOptimizer {
    pub fn new() -> Self { Self }

    // ── Per-asset helpers ─────────────────────────────────────────────────────

    /// Compute optimal fraction and ELG for one asset from its ensemble distribution.
    ///
    /// Uses the per-asset estimated_tc_bps from the `AssetDistribution` rather
    /// than the portfolio-level `transaction_cost_bps`, enabling asset-specific
    /// cost modelling (e.g. wider spreads on illiquid assets).
    fn optimize_single_dist(
        ad: &AssetDistribution,
        constraints: &PortfolioConstraints,
    ) -> (f64, f64) {
        let mu = ad.ensemble.predictive_mean;
        let sigma2 = ad.ensemble.predictive_variance();

        let denom = sigma2 + mu * mu;
        let kelly_full = if denom > 1e-15 { mu / denom } else { 0.0 };
        let kelly_frac = constraints.kelly_fraction * kelly_full;

        let tc_fraction = ad.estimated_tc_bps * 1e-4;
        let tc_adj = kelly_frac
            - constraints.turnover_penalty
            * (kelly_frac - ad.current_fraction).signum()
            * tc_fraction;
        let f = tc_adj.clamp(-constraints.max_position_fraction, constraints.max_position_fraction);

        let elg = f * mu - 0.5 * f * f * (sigma2 + mu * mu);
        (f, elg)
    }

    /// Compute optimal fraction from a raw `EnsembleDistribution`.
    ///
    /// Used for the backtest compatibility path where `AssetDistribution` is not
    /// available (uses the portfolio-level TC estimate).
    fn optimize_single(
        dist: &EnsembleDistribution,
        current_fraction: f64,
        constraints: &PortfolioConstraints,
    ) -> (f64, f64) {
        let mu = dist.predictive_mean;
        let sigma2 = dist.predictive_variance();

        let denom = sigma2 + mu * mu;
        let kelly_full = if denom > 1e-15 { mu / denom } else { 0.0 };
        let kelly_frac = constraints.kelly_fraction * kelly_full;

        let tc = constraints.transaction_cost_bps * 1e-4;
        let tc_adj = kelly_frac
            - constraints.turnover_penalty * (kelly_frac - current_fraction).signum() * tc;
        let f = tc_adj.clamp(-constraints.max_position_fraction, constraints.max_position_fraction);
        let elg = f * mu - 0.5 * f * f * (sigma2 + mu * mu);

        (f, elg)
    }

    /// Multi-period DP: backward induction on a 41-point position grid.
    fn dp_optimal_fraction(
        dist: &EnsembleDistribution,
        current_fraction: f64,
        constraints: &PortfolioConstraints,
        horizon: usize,
    ) -> f64 {
        if horizon <= 1 {
            return Self::optimize_single(dist, current_fraction, constraints).0;
        }

        let n_grid = 41usize;
        let fmax = constraints.max_position_fraction;
        let grid: Vec<f64> = (0..n_grid)
            .map(|i| -fmax + 2.0 * fmax * i as f64 / (n_grid - 1) as f64)
            .collect();

        let mut v_next: Vec<f64> = grid
            .iter()
            .map(|&f| {
                let (f_opt, elg) = Self::optimize_single(dist, f, constraints);
                let tc_cost = constraints.turnover_penalty * (f_opt - f).abs();
                elg - tc_cost
            })
            .collect();

        for _ in 1..horizon {
            let mut v_curr = vec![0.0f64; n_grid];
            for (i, &f) in grid.iter().enumerate() {
                let best_v = grid
                    .iter()
                    .zip(&v_next)
                    .map(|(&f_next, &v_n)| {
                        let mu = dist.predictive_mean;
                        let s2 = dist.predictive_variance();
                        let one_step = f_next * mu - 0.5 * f_next * f_next * (s2 + mu * mu);
                        let tc = constraints.turnover_penalty * (f_next - f).abs();
                        one_step - tc + v_n
                    })
                    .fold(f64::NEG_INFINITY, f64::max);
                v_curr[i] = best_v;
            }
            v_next = v_curr;
        }

        let current_idx = ((current_fraction + fmax) / (2.0 * fmax) * (n_grid - 1) as f64)
            .round()
            .clamp(0.0, (n_grid - 1) as f64) as usize;

        grid.iter()
            .zip(&v_next)
            .map(|(&f_next, &v_n)| {
                let mu = dist.predictive_mean;
                let s2 = dist.predictive_variance();
                let one_step = f_next * mu - 0.5 * f_next * f_next * (s2 + mu * mu);
                let tc = constraints.turnover_penalty * (f_next - grid[current_idx]).abs();
                (f_next, one_step - tc + v_n)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(f, _)| f)
            .unwrap_or(0.0)
    }

    // ── Subset selection ──────────────────────────────────────────────────────

    /// Exhaustive subset search over all C(n, k) subsets for k = 1..=max_k.
    ///
    /// Objective: maximise Σ_i [ELG_i − turnover_i·tc_i/10000] subject to
    /// Σ_i [turnover_i·tc_bps_i] ≤ max_tc_bps.
    ///
    /// Only called when n_candidates ≤ 10 (at most C(10,3) = 120 evaluations).
    fn best_subset_exhaustive(
        candidates: &[AssetCandidate<'_>],
        max_k: usize,
        max_tc_bps: f64,
    ) -> Vec<usize> {
        let n = candidates.len();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_set: Vec<usize> = vec![];

        for k in 1..=max_k {
            for combo in enumerate_combinations(n, k) {
                let total_tc: f64 = combo.iter().map(|&i| candidates[i].tc_bps).sum();
                if total_tc > max_tc_bps {
                    continue;
                }
                let score: f64 = combo
                    .iter()
                    .map(|&i| candidates[i].net_score())
                    .sum();
                if score > best_score {
                    best_score = score;
                    best_set = combo;
                }
            }
        }
        best_set
    }

    /// Greedy subset selection for large candidate sets (n > 10).
    ///
    /// Ranks candidates by net_score descending, accumulates until max_k is
    /// reached or adding the next asset would exceed the TC budget.
    fn greedy_select(
        candidates: &[AssetCandidate<'_>],
        max_k: usize,
        max_tc_bps: f64,
    ) -> Vec<usize> {
        let mut ranked: Vec<usize> = (0..candidates.len()).collect();
        ranked.sort_by(|&a, &b| {
            candidates[b]
                .net_score()
                .partial_cmp(&candidates[a].net_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut selected = vec![];
        let mut cumulative_tc = 0.0f64;
        for &idx in ranked.iter().take(max_k) {
            let next_tc = cumulative_tc + candidates[idx].tc_bps;
            if next_tc > max_tc_bps {
                continue; // skip this asset (too expensive), keep trying others
            }
            cumulative_tc = next_tc;
            selected.push(idx);
        }
        selected
    }
}

// ── AssetCandidate ─────────────────────────────────────────────────────────────

/// Internal: one candidate asset considered by the joint optimizer.
struct AssetCandidate<'a> {
    ad: &'a AssetDistribution,
    /// Optimal signed position fraction (clipped to constraints).
    f: f64,
    /// Per-period expected log-growth at the optimal fraction.
    elg: f64,
    /// Transaction cost for this leg: |Δf| × estimated_tc_bps (in bps units).
    tc_bps: f64,
}

impl<'a> AssetCandidate<'a> {
    /// Net score used for subset ranking/selection.
    ///
    /// Converts TC from bps to fraction before subtracting so both terms are
    /// dimensionless (expected log-growth units per bar).
    fn net_score(&self) -> f64 {
        self.elg - self.tc_bps * 1e-4
    }
}

// ── Combination enumeration ───────────────────────────────────────────────────

fn enumerate_combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut combo = Vec::with_capacity(k);
    comb_helper(0, n, k, &mut combo, &mut result);
    result
}

fn comb_helper(
    start: usize,
    n: usize,
    k: usize,
    combo: &mut Vec<usize>,
    result: &mut Vec<Vec<usize>>,
) {
    if combo.len() == k {
        result.push(combo.clone());
        return;
    }
    let remaining = k - combo.len();
    let end = n.saturating_sub(remaining);
    for i in start..=end {
        combo.push(i);
        comb_helper(i + 1, n, k, combo, result);
        combo.pop();
    }
}

// ── Regime helpers ────────────────────────────────────────────────────────────

fn regime_min_rr(regime: &str) -> f64 {
    match regime {
        "high_vol" => 3.0,
        "trending" => 2.0,
        _ => 1.5,
    }
}

fn timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ── PortfolioOptimizer impl ───────────────────────────────────────────────────

impl Default for KellyOptimizer {
    fn default() -> Self { Self::new() }
}

impl PortfolioOptimizer for KellyOptimizer {
    /// Joint multi-asset optimization producing a `ComboOrder`.
    fn optimize_combo(
        &self,
        asset_dists: &[AssetDistribution],
        portfolio: &PortfolioState,
        constraints: &PortfolioConstraints,
        combo_config: &ComboConfig,
    ) -> Option<ComboOrder> {
        if asset_dists.is_empty() || portfolio.nav < 1.0 {
            return None;
        }

        // ── Step 1: per-asset Kelly fractions ────────────────────────────────
        let candidates: Vec<AssetCandidate<'_>> = asset_dists
            .iter()
            .filter(|ad| {
                ad.ensemble.is_confident()
                    && ad.ensemble.confidence >= constraints.min_confidence
                    && ad.ensemble.directional_edge().abs() >= 0.05
            })
            .filter_map(|ad| {
                let (f, elg) = Self::optimize_single_dist(ad, constraints);
                if f.abs() < 1e-4 {
                    return None;
                }
                if elg < combo_config.min_leg_elg {
                    return None;
                }
                let turnover = (f - ad.current_fraction).abs();
                let tc_bps = turnover * ad.estimated_tc_bps;
                Some(AssetCandidate { ad, f, elg, tc_bps })
            })
            .collect();

        if candidates.is_empty() {
            return None;
        }

        // ── Step 2: select best subset (cardinality + TC budget) ─────────────
        let n = candidates.len();
        let k = combo_config.max_assets.min(n);
        let max_tc = combo_config.max_total_tc_bps;

        let selected_indices = if combo_config.enumerate_subsets && n <= 10 {
            Self::best_subset_exhaustive(&candidates, k, max_tc)
        } else {
            Self::greedy_select(&candidates, k, max_tc)
        };

        if selected_indices.is_empty() {
            return None;
        }

        // Optionally trim to max_traded_assets (assets with non-trivial turnover).
        let mut traded_count = 0usize;
        let selected_indices: Vec<usize> = selected_indices
            .into_iter()
            .filter(|&i| {
                if (candidates[i].f - candidates[i].ad.current_fraction).abs() > 1e-4 {
                    if traded_count >= combo_config.max_traded_assets {
                        return false;
                    }
                    traded_count += 1;
                }
                true
            })
            .collect();

        if selected_indices.is_empty() {
            return None;
        }

        // ── Step 3: portfolio constraint scaling (gross + net) ────────────────
        let gross_sum: f64 = selected_indices.iter().map(|&i| candidates[i].f.abs()).sum();
        let net_sum: f64 = selected_indices.iter().map(|&i| candidates[i].f).sum();

        let scale_gross = if gross_sum > constraints.max_gross_exposure {
            constraints.max_gross_exposure / gross_sum
        } else {
            1.0
        };
        let scale_net = if net_sum.abs() > constraints.max_net_exposure {
            constraints.max_net_exposure / net_sum.abs()
        } else {
            1.0
        };
        let scale = scale_gross.min(scale_net);

        // ── Step 4: build legs ────────────────────────────────────────────────
        let mut legs = Vec::with_capacity(selected_indices.len());
        let mut total_elg = 0.0f64;
        let mut total_tc = 0.0f64;

        for &idx in &selected_indices {
            let cand = &candidates[idx];
            let f_scaled = cand.f * scale;
            let direction = if f_scaled >= 0.0 { Direction::Long } else { Direction::Short };
            let fraction = f_scaled.abs().min(constraints.max_position_fraction);
            let leverage = (fraction * constraints.max_leverage).clamp(1.0, constraints.max_leverage);

            let entry = cand.ad.entry_price;
            let (stop_loss, take_profit) = if entry > 0.0 {
                let stop_dist = cand.ad.stop_distance_fraction() * entry;
                let min_rr = regime_min_rr(cand.ad.regime_label());
                match direction {
                    Direction::Long => (entry - stop_dist, entry + stop_dist * min_rr),
                    Direction::Short => (entry + stop_dist, entry - stop_dist * min_rr),
                }
            } else {
                (0.0, 0.0)
            };

            total_elg += cand.elg;
            total_tc += cand.tc_bps;

            legs.push(OrderLeg {
                asset: cand.ad.asset.clone(),
                direction,
                position_fraction: fraction,
                position_size_usd: portfolio.nav * fraction,
                leverage,
                entry_price: entry,
                stop_loss,
                take_profit,
                expected_log_growth: cand.elg,
                leg_tc_bps: cand.tc_bps,
                confidence: cand.ad.ensemble.confidence,
                regime_label: cand.ad.regime_label().to_string(),
            });
        }

        let gross: f64 = legs.iter().map(|l| l.position_fraction).sum();
        let net: f64 = legs.iter().map(|l| {
            if l.is_long() { l.position_fraction } else { -l.position_fraction }
        }).sum();

        Some(ComboOrder {
            combo_id: Uuid::new_v4().to_string(),
            generated_at_ms: timestamp_ms(),
            legs,
            total_expected_log_growth: total_elg,
            total_tc_bps: total_tc,
            gross_exposure_fraction: gross,
            net_exposure_fraction: net,
        })
    }

    /// Multi-period DP: uses backward induction per asset, then joint portfolio
    /// constraints applied to the DP-optimal fractions.
    fn optimize_dp(
        &self,
        distributions: &HashMap<String, EnsembleDistribution>,
        portfolio: &PortfolioState,
        constraints: &PortfolioConstraints,
        horizon: usize,
    ) -> Vec<TradeDecision> {
        if distributions.is_empty() || portfolio.nav < 1.0 {
            return vec![];
        }

        let mut candidates: Vec<(String, f64, f64, &EnsembleDistribution)> = Vec::new();

        for (asset, dist) in distributions {
            if !dist.is_confident() || dist.directional_edge().abs() < 0.05 {
                continue;
            }
            let current_pos = portfolio.positions.get(asset);
            let current_fraction = current_pos
                .map(|p| {
                    let sign = if p.direction.is_long() { 1.0 } else { -1.0 };
                    sign * p.size_usd / portfolio.nav.max(1.0)
                })
                .unwrap_or(0.0);

            let f = Self::dp_optimal_fraction(dist, current_fraction, constraints, horizon);
            if f.abs() < 1e-4 { continue; }

            let (_, elg) = Self::optimize_single(dist, current_fraction, constraints);
            candidates.push((asset.clone(), f, elg, dist));
        }

        if candidates.is_empty() {
            return vec![];
        }

        // Portfolio constraint scaling.
        let gross_sum: f64 = candidates.iter().map(|(_, f, _, _)| f.abs()).sum();
        let scale_gross = if gross_sum > constraints.max_gross_exposure {
            constraints.max_gross_exposure / gross_sum
        } else { 1.0 };
        let net_sum: f64 = candidates.iter().map(|(_, f, _, _)| *f).sum();
        let scale_net = if net_sum.abs() > constraints.max_net_exposure {
            constraints.max_net_exposure / net_sum.abs()
        } else { 1.0 };
        let scale = scale_gross.min(scale_net);

        let mut decisions: Vec<TradeDecision> = candidates
            .into_iter()
            .map(|(asset, f, elg, dist)| {
                let f_s = f * scale;
                let direction = if f_s >= 0.0 { Direction::Long } else { Direction::Short };
                let fraction = f_s.abs().min(constraints.max_position_fraction);
                let leverage = (fraction * constraints.max_leverage).clamp(1.0, constraints.max_leverage);
                TradeDecision {
                    asset,
                    direction,
                    position_fraction: fraction,
                    leverage,
                    position_size_usd: portfolio.nav * fraction,
                    expected_log_growth: elg,
                    confidence: dist.confidence,
                    regime_label: dist.regime_description().to_string(),
                }
            })
            .collect();

        decisions.sort_by(|a, b| b.expected_log_growth.partial_cmp(&a.expected_log_growth).unwrap());
        decisions
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dist(p_up: f64, p_down: f64, mu: f64, sigma: f64) -> EnsembleDistribution {
        EnsembleDistribution {
            p_up,
            p_down,
            predictive_mean: mu,
            predictive_std: sigma,
            confidence: 0.7,
            weights: vec![],
        }
    }

    fn make_asset_dist(asset: &str, mu: f64, sigma: f64, tc_bps: f64) -> AssetDistribution {
        AssetDistribution {
            asset: asset.to_string(),
            ensemble: make_dist(if mu > 0.0 { 0.6 } else { 0.2 }, if mu < 0.0 { 0.6 } else { 0.2 }, mu, sigma),
            current_fraction: 0.0,
            entry_price: 100.0,
            estimated_tc_bps: tc_bps,
        }
    }

    fn default_constraints() -> PortfolioConstraints {
        PortfolioConstraints::from_risk_config(3.0, 0.30, 0.5)
    }

    // ── Single-asset helpers ─────────────────────────────────────────────────

    #[test]
    fn kelly_long_for_positive_drift() {
        let constraints = default_constraints();
        let dist = make_dist(0.6, 0.2, 0.005, 0.01);
        let (f, elg) = KellyOptimizer::optimize_single(&dist, 0.0, &constraints);
        assert!(f > 0.0, "Expected long position, got f={}", f);
        assert!(elg > 0.0, "Expected positive ELG, got {}", elg);
    }

    #[test]
    fn kelly_short_for_negative_drift() {
        let constraints = default_constraints();
        let dist = make_dist(0.2, 0.6, -0.005, 0.01);
        let (f, _) = KellyOptimizer::optimize_single(&dist, 0.0, &constraints);
        assert!(f < 0.0, "Expected short position, got f={}", f);
    }

    // ── Backtest-compat optimize() via default trait impl ────────────────────

    #[test]
    fn gross_exposure_capped() {
        let constraints = default_constraints();
        let opt = KellyOptimizer::new();
        let portfolio = PortfolioState::new(10_000.0);

        let mut dists = HashMap::new();
        for sym in ["BTC", "ETH", "SOL", "BNB", "DOGE"] {
            dists.insert(sym.to_string(), make_dist(0.7, 0.1, 0.01, 0.01));
        }
        let decisions = opt.optimize(&dists, &portfolio, &constraints);
        let gross: f64 = decisions.iter().map(|d| d.position_fraction).sum();
        assert!(
            gross <= constraints.max_gross_exposure + 1e-9,
            "gross={} exceeds limit={}",
            gross,
            constraints.max_gross_exposure
        );
    }

    // ── Joint combo optimization ─────────────────────────────────────────────

    #[test]
    fn combo_selects_best_subset() {
        let opt = KellyOptimizer::new();
        let portfolio = PortfolioState::new(100_000.0);
        let constraints = default_constraints();
        let combo_config = ComboConfig { max_assets: 2, max_total_tc_bps: 100.0, ..Default::default() };

        let asset_dists = vec![
            make_asset_dist("BTC", 0.01, 0.01, 5.0),  // strong positive
            make_asset_dist("ETH", -0.01, 0.01, 5.0), // strong negative (short)
            make_asset_dist("SOL", 0.001, 0.05, 5.0), // weak / high vol → low ELG
        ];

        let combo = opt.optimize_combo(&asset_dists, &portfolio, &constraints, &combo_config)
            .expect("Expected combo order");

        assert!(combo.n_legs() <= 2, "Should have at most 2 legs");
        // BTC (long) and ETH (short) should dominate due to higher |ELG|.
        let assets: Vec<&str> = combo.legs.iter().map(|l| l.asset.as_str()).collect();
        assert!(
            assets.contains(&"BTC") && assets.contains(&"ETH"),
            "Expected BTC and ETH in combo, got {:?}",
            assets
        );
        // Directions should match drift signs.
        let btc_leg = combo.legs.iter().find(|l| l.asset == "BTC").unwrap();
        let eth_leg = combo.legs.iter().find(|l| l.asset == "ETH").unwrap();
        assert!(btc_leg.is_long(), "BTC with positive drift should be long");
        assert!(!eth_leg.is_long(), "ETH with negative drift should be short");
    }

    #[test]
    fn combo_enforces_tc_budget() {
        let opt = KellyOptimizer::new();
        let portfolio = PortfolioState::new(100_000.0);
        let constraints = default_constraints();
        // Very tight TC budget: only one asset can fit.
        let combo_config = ComboConfig {
            max_assets: 3,
            max_total_tc_bps: 3.0,  // tight: single asset at 30% position × 5bps = 1.5bps → fits 2
            max_traded_assets: 3,
            min_leg_elg: 0.0,
            enumerate_subsets: true,
        };

        // All assets have high TC (10bps). Only the best can fit within 3bps budget.
        let asset_dists = vec![
            make_asset_dist("BTC", 0.01, 0.01, 10.0),
            make_asset_dist("ETH", 0.009, 0.01, 10.0),
            make_asset_dist("SOL", 0.008, 0.01, 10.0),
        ];

        let combo = opt.optimize_combo(&asset_dists, &portfolio, &constraints, &combo_config)
            .expect("Expected at least one leg");

        // With tc_bps=10 and turnover≈0.30 (max_position_fraction), each leg costs ~3bps.
        // Budget=3bps → at most 1 leg.
        assert!(
            combo.n_legs() <= 1,
            "TC budget should limit to 1 leg, got {} legs (total_tc={:.2})",
            combo.n_legs(),
            combo.total_tc_bps
        );
        assert!(
            combo.total_tc_bps <= 3.0 + 1e-9,
            "Total TC {:.2} exceeds budget 3.0",
            combo.total_tc_bps
        );
    }

    #[test]
    fn combo_respects_cardinality() {
        let opt = KellyOptimizer::new();
        let portfolio = PortfolioState::new(100_000.0);
        let constraints = default_constraints();
        let combo_config = ComboConfig { max_assets: 2, max_total_tc_bps: 1000.0, ..Default::default() };

        let asset_dists: Vec<AssetDistribution> = ["BTC", "ETH", "SOL", "BNB"]
            .iter()
            .map(|&s| make_asset_dist(s, 0.005, 0.01, 1.0))
            .collect();

        let combo = opt.optimize_combo(&asset_dists, &portfolio, &constraints, &combo_config)
            .expect("Expected combo order");

        assert!(
            combo.n_legs() <= 2,
            "Cardinality limit of 2 violated: got {} legs",
            combo.n_legs()
        );
    }

    #[test]
    fn combo_net_exposure_bounded() {
        let opt = KellyOptimizer::new();
        let portfolio = PortfolioState::new(100_000.0);
        // max_net_exposure = 3.0 * 0.6 = 1.8.
        let constraints = default_constraints();
        let combo_config = ComboConfig { max_assets: 5, max_total_tc_bps: 1000.0, ..Default::default() };

        // All long → net = gross; should be scaled to max_net_exposure.
        let asset_dists: Vec<AssetDistribution> = ["A", "B", "C", "D", "E"]
            .iter()
            .map(|&s| make_asset_dist(s, 0.02, 0.005, 1.0))
            .collect();

        let combo = opt.optimize_combo(&asset_dists, &portfolio, &constraints, &combo_config)
            .expect("Expected combo");

        assert!(
            combo.net_exposure_fraction.abs() <= constraints.max_net_exposure + 1e-9,
            "net_exposure={:.3} exceeds max={:.3}",
            combo.net_exposure_fraction,
            constraints.max_net_exposure
        );
        assert!(
            combo.gross_exposure_fraction <= constraints.max_gross_exposure + 1e-9,
            "gross_exposure={:.3} exceeds max={:.3}",
            combo.gross_exposure_fraction,
            constraints.max_gross_exposure
        );
    }
}
