//! Fractional Kelly portfolio optimizer with portfolio-level constraints.
//!
//! ## Method
//! For each asset, maximize E[log(1 + f·r)] subject to:
//!   - position limits
//!   - portfolio gross/net exposure
//!   - leverage cap
//!   - transaction cost penalty
//!
//! E[log(1 + f·r)] is evaluated by a second-order Taylor expansion around f=0:
//!   ≈ f·μ − ½·f²·(σ² + μ²)
//! where (μ, σ) are the ensemble predictive mean and std.
//!
//! The unconstrained Kelly fraction is f* = μ / (σ² + μ²).
//! Fractional Kelly applies: f = kelly_fraction · f*.
//! Constraints clip f to feasible bounds.
//!
//! ## Transaction costs
//! Turnover penalty: λ_tc · |f_new − f_prev| is subtracted from the objective,
//! reducing position flipping.  f_prev is read from the current portfolio state.
//!
//! ## Portfolio-level constraint propagation
//! After per-asset optimal fractions are computed, if the gross exposure sum
//! exceeds the limit, all positions are uniformly scaled down.
//!
//! ## Multi-period DP extension
//! `optimize_dp` computes a backward-induction solution over `horizon` steps,
//! using the one-period model as the reward function at each stage.

use std::collections::HashMap;

use crate::{
    ensemble::EnsembleDistribution,
    risk::manager::PortfolioState,
    signals::generator::Direction,
};

use super::{PortfolioConstraints, PortfolioOptimizer, TradeDecision};

// ─────────────────────────────────────────────────────────────────────────────

pub struct KellyOptimizer;

impl KellyOptimizer {
    pub fn new() -> Self {
        Self
    }

    // ── Core single-asset optimization ───────────────────────────────────────

    /// Compute optimal fractional position for one asset.
    ///
    /// Returns (optimal_fraction, expected_log_growth).
    /// fraction > 0 → long, fraction < 0 → short.
    fn optimize_single(
        dist: &EnsembleDistribution,
        current_fraction: f64,
        constraints: &PortfolioConstraints,
    ) -> (f64, f64) {
        let mu = dist.predictive_mean;
        let sigma2 = dist.predictive_variance();

        // Kelly fraction: f* = μ / (σ² + μ²).
        let denom = sigma2 + mu * mu;
        let kelly_full = if denom > 1e-15 { mu / denom } else { 0.0 };
        let kelly_frac = constraints.kelly_fraction * kelly_full;

        // Transaction cost adjustment: subtract penalty from target.
        let tc = constraints.transaction_cost_bps * 1e-4; // convert bps to fraction
        let tc_adj = kelly_frac - constraints.turnover_penalty * (kelly_frac - current_fraction).signum() * tc;

        // Clip to position constraint.
        let f = tc_adj.clamp(-constraints.max_position_fraction, constraints.max_position_fraction);

        // Expected log-growth (second-order approx).
        let elg = f * mu - 0.5 * f * f * (sigma2 + mu * mu);

        (f, elg)
    }

    /// Multi-period DP: maximize cumulative expected log-growth over `horizon` bars.
    ///
    /// Uses backward induction with the one-period model as terminal value.
    /// The DP state is the fractional position; transition is controlled by the
    /// optimizer at each step.
    ///
    /// For tractability, the distribution is assumed constant over the horizon
    /// (recalibrated live). The DP mainly captures the effect of turnover costs
    /// over the holding period.
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

        // V[i] = value-to-go from position grid[i] at the current time.
        let mut v_next: Vec<f64> = grid
            .iter()
            .map(|&f| {
                let (f_opt, elg) = Self::optimize_single(dist, f, constraints);
                let tc_cost = constraints.turnover_penalty * (f_opt - f).abs();
                elg - tc_cost
            })
            .collect();

        // Backward induction: terminal value for remaining steps.
        for _ in 1..horizon {
            let mut v_curr = vec![0.0f64; n_grid];
            for (i, &f) in grid.iter().enumerate() {
                // Find best action from position f.
                let best_v = grid
                    .iter()
                    .zip(&v_next)
                    .map(|(&f_next, &v_n)| {
                        let mu = dist.predictive_mean;
                        let sigma2 = dist.predictive_variance();
                        let one_step = f_next * mu - 0.5 * f_next * f_next * (sigma2 + mu * mu);
                        let tc = constraints.turnover_penalty * (f_next - f).abs();
                        one_step - tc + v_n
                    })
                    .fold(f64::NEG_INFINITY, f64::max);
                v_curr[i] = best_v;
            }
            v_next = v_curr;
        }

        // Best starting action from current_fraction.
        let current_idx = ((current_fraction + fmax) / (2.0 * fmax) * (n_grid - 1) as f64)
            .round()
            .clamp(0.0, (n_grid - 1) as f64) as usize;

        // Find the grid action that maximises first-step value + continuation value.
        grid.iter()
            .zip(&v_next)
            .map(|(&f_next, &v_n)| {
                let mu = dist.predictive_mean;
                let sigma2 = dist.predictive_variance();
                let one_step = f_next * mu - 0.5 * f_next * f_next * (sigma2 + mu * mu);
                let tc = constraints.turnover_penalty * (f_next - grid[current_idx]).abs();
                (f_next, one_step - tc + v_n)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(f, _)| f)
            .unwrap_or(0.0)
    }
}

impl Default for KellyOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl PortfolioOptimizer for KellyOptimizer {
    fn optimize(
        &self,
        distributions: &HashMap<String, EnsembleDistribution>,
        portfolio: &PortfolioState,
        constraints: &PortfolioConstraints,
    ) -> Vec<TradeDecision> {
        self.optimize_dp(distributions, portfolio, constraints, 1)
    }

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

        // Per-asset: compute unconstrained optimal fractions.
        let mut candidates: Vec<(String, f64, f64, &EnsembleDistribution)> = Vec::new();

        for (asset, dist) in distributions {
            if !dist.is_confident() || dist.confidence < constraints.min_confidence {
                continue;
            }

            // Skip if edge is negligible.
            if dist.directional_edge().abs() < 0.05 {
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
            if f.abs() < 1e-4 {
                continue; // No meaningful trade.
            }

            let (_, elg) = Self::optimize_single(dist, current_fraction, constraints);
            candidates.push((asset.clone(), f, elg, dist));
        }

        if candidates.is_empty() {
            return vec![];
        }

        // Portfolio-level constraint: scale down if gross exposure exceeded.
        let gross_sum: f64 = candidates.iter().map(|(_, f, _, _)| f.abs()).sum();
        let scale = if gross_sum > constraints.max_gross_exposure {
            constraints.max_gross_exposure / gross_sum
        } else {
            1.0
        };

        // Net exposure check.
        let net_sum: f64 = candidates.iter().map(|(_, f, _, _)| *f).sum();
        let net_scale = if net_sum.abs() > constraints.max_net_exposure {
            constraints.max_net_exposure / net_sum.abs()
        } else {
            1.0
        };
        let scale = scale.min(net_scale);

        // Build TradeDecision objects.
        let mut decisions: Vec<TradeDecision> = candidates
            .into_iter()
            .map(|(asset, f, elg, dist)| {
                let f_scaled = f * scale;
                let direction = if f_scaled >= 0.0 { Direction::Long } else { Direction::Short };
                let fraction = f_scaled.abs().min(constraints.max_position_fraction);
                let leverage = fraction
                    .mul_add(constraints.max_leverage, 0.0)
                    .clamp(1.0, constraints.max_leverage);

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

        // Sort by expected log-growth descending (best opportunities first).
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

    #[test]
    fn kelly_long_for_positive_drift() {
        let constraints = PortfolioConstraints::from_risk_config(3.0, 0.30, 0.5);
        let opt = KellyOptimizer::new();
        let dist = make_dist(0.6, 0.2, 0.005, 0.01);
        let (f, elg) = KellyOptimizer::optimize_single(&dist, 0.0, &constraints);
        assert!(f > 0.0, "Expected long position, got f={}", f);
        assert!(elg > 0.0, "Expected positive ELG, got {}", elg);
    }

    #[test]
    fn kelly_short_for_negative_drift() {
        let constraints = PortfolioConstraints::from_risk_config(3.0, 0.30, 0.5);
        let dist = make_dist(0.2, 0.6, -0.005, 0.01);
        let (f, _) = KellyOptimizer::optimize_single(&dist, 0.0, &constraints);
        assert!(f < 0.0, "Expected short position, got f={}", f);
    }

    #[test]
    fn gross_exposure_capped() {
        use crate::risk::manager::PortfolioState;
        let constraints = PortfolioConstraints::from_risk_config(3.0, 0.30, 0.5);
        let opt = KellyOptimizer::new();
        let portfolio = PortfolioState::new(10_000.0);

        let mut dists = HashMap::new();
        for sym in ["BTC", "ETH", "SOL", "BNB", "DOGE"] {
            dists.insert(sym.to_string(), make_dist(0.7, 0.1, 0.01, 0.01));
        }
        let decisions = opt.optimize(&dists, &portfolio, &constraints);
        let gross: f64 = decisions.iter().map(|d| d.position_fraction).sum();
        assert!(gross <= constraints.max_gross_exposure + 1e-9,
            "gross={} exceeds limit={}", gross, constraints.max_gross_exposure);
    }
}
