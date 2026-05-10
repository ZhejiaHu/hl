//! Portfolio optimizer traits and shared types.
//!
//! ## Per-asset distribution estimate
//! `AssetDistribution` bundles everything computed before optimization: the fused
//! ensemble distribution, current portfolio position, entry price estimate, and a
//! transaction cost proxy. This is the single input type to the joint optimizer.
//!
//! ## Joint multi-asset optimization
//! `PortfolioOptimizer::optimize_combo` solves the portfolio problem across all
//! assets simultaneously and returns a `ComboOrder` — a set of coordinated legs,
//! e.g. "long 0.20 BTC + short 0.15 ETH". Per-asset sizing and portfolio-level
//! constraint enforcement happen in one pass, replacing the old pattern of
//! independent per-asset Kelly + ad-hoc gross-exposure scaling.
//!
//! ## Optimization problem
//!   maximize  Σ_i [f_i·μ_i − ½·f_i²·(σ_i²+μ_i²)] − λ_tc · Σ_i |Δf_i|·tc_i
//!   subject to:
//!     |f_i|              ≤ max_position_fraction      (per-asset)
//!     Σ_i |f_i|          ≤ max_gross_exposure         (portfolio)
//!     |Σ_i f_i|          ≤ max_net_exposure            (portfolio)
//!     card({i: |f_i|>0}) ≤ max_assets                 (cardinality)
//!     Σ_i |Δf_i|·tc_i    ≤ max_total_tc_bps           (TC budget)

use std::collections::HashMap;

use crate::{
    config::Config,
    ensemble::EnsembleDistribution,
    risk::manager::PortfolioState,
    signals::generator::Direction,
};

// ── Per-asset distribution estimate ──────────────────────────────────────────

/// Everything computed before portfolio optimization, bundled per asset.
///
/// Built once per tick per asset by the main loop (or backtest engine) after
/// the estimation → ensemble pipeline has run. The joint optimizer reads a
/// slice of these and produces a `ComboOrder`.
#[derive(Debug, Clone)]
pub struct AssetDistribution {
    pub asset: String,
    /// Fused ensemble distribution (the regime representation, replaces HMM state).
    pub ensemble: EnsembleDistribution,
    /// Signed current position fraction (positive=long, negative=short, 0=flat).
    /// Units: fraction of NAV.
    pub current_fraction: f64,
    /// Kalman-filtered price used as the execution price target for this tick.
    pub entry_price: f64,
    /// One-way transaction cost estimate in basis points (spread + impact proxy).
    /// Set from the live order book each tick; used in both objective and TC budget.
    pub estimated_tc_bps: f64,
    /// 95th-percentile CVaR (absolute value) from the fitted return distribution.
    pub cvar_95: f64,
    /// 99th-percentile CVaR (absolute value) from the fitted return distribution.
    pub cvar_99: f64,
}

impl AssetDistribution {
    #[inline]
    pub fn predictive_mean(&self) -> f64 {
        self.ensemble.predictive_mean
    }

    #[inline]
    pub fn predictive_std(&self) -> f64 {
        self.ensemble.predictive_std
    }

    #[inline]
    pub fn regime_label(&self) -> &'static str {
        self.ensemble.regime_description()
    }

    /// Stop distance as a fraction of price: 2 × max(predictive_std, CVaR_95, 0.1%).
    pub fn stop_distance_fraction(&self) -> f64 {
        2.0 * self.predictive_std().max(self.cvar_95).max(0.001)
    }
}

// ── Combo configuration ───────────────────────────────────────────────────────

/// Configuration for the joint multi-asset portfolio optimizer.
///
/// All fields are populated from environment variables via `from_config` —
/// no hardcoded asset-specific values.
#[derive(Debug, Clone)]
pub struct ComboConfig {
    /// Maximum number of assets (legs) in the output combo order.
    /// Cardinality constraint: card({i: |f_i|>0}) ≤ max_assets.
    pub max_assets: usize,
    /// Total transaction cost budget across all legs in basis points.
    /// Assets are dropped (worst ELG/TC ratio first) to fit within this budget.
    pub max_total_tc_bps: f64,
    /// Maximum number of assets whose position changes this tick.
    /// Allows holding up to max_assets while limiting new order submissions to
    /// max_traded_assets per cycle (same as max_assets by default).
    pub max_traded_assets: usize,
    /// Minimum expected log-growth for a leg to be included (pre-TC filter).
    pub min_leg_elg: f64,
    /// When true and n_candidates ≤ 10: enumerate all C(n,k) subsets for k=1..max_assets
    /// to find the globally optimal subset. Falls back to greedy for larger n.
    pub enumerate_subsets: bool,
}

impl Default for ComboConfig {
    fn default() -> Self {
        Self {
            max_assets: 3,
            max_total_tc_bps: 25.0,
            max_traded_assets: 3,
            min_leg_elg: 0.0,
            enumerate_subsets: true,
        }
    }
}

impl ComboConfig {
    pub fn from_config(config: &Config) -> Self {
        Self {
            max_assets: config.max_combo_assets,
            max_total_tc_bps: config.max_combo_tc_bps,
            max_traded_assets: config.max_combo_assets,
            min_leg_elg: 0.0,
            enumerate_subsets: config.enumerate_combo_subsets,
        }
    }
}

// ── Output: combo order ───────────────────────────────────────────────────────

/// One leg of a joint combo order: a fully specified single-asset trade.
#[derive(Debug, Clone)]
pub struct OrderLeg {
    pub asset: String,
    pub direction: Direction,
    /// Fractional position size as share of NAV ∈ (0, max_position_fraction].
    pub position_fraction: f64,
    /// Absolute notional in USD (= NAV × position_fraction).
    pub position_size_usd: f64,
    pub leverage: f64,
    pub entry_price: f64,
    pub stop_loss: f64,
    pub take_profit: f64,
    /// Expected log-growth contribution of this leg under the ensemble model.
    pub expected_log_growth: f64,
    /// Transaction cost consumed by this leg: turnover × estimated_tc_bps.
    pub leg_tc_bps: f64,
    pub confidence: f64,
    pub regime_label: String,
}

impl OrderLeg {
    pub fn is_long(&self) -> bool {
        matches!(self.direction, Direction::Long)
    }

    pub fn risk_reward(&self) -> f64 {
        let gain = (self.take_profit - self.entry_price).abs();
        let loss = (self.stop_loss - self.entry_price).abs();
        if loss < 1e-10 { 0.0 } else { gain / loss }
    }

    pub fn stop_distance_pct(&self) -> f64 {
        if self.entry_price < 1e-10 { return 0.0; }
        (self.stop_loss - self.entry_price).abs() / self.entry_price
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "asset": self.asset,
            "direction": self.direction.to_string(),
            "position_fraction": self.position_fraction,
            "position_size_usd": self.position_size_usd,
            "leverage": self.leverage,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "expected_log_growth": self.expected_log_growth,
            "leg_tc_bps": self.leg_tc_bps,
            "confidence": self.confidence,
            "regime_label": self.regime_label,
            "risk_reward": self.risk_reward(),
        })
    }
}

/// A fully computed joint portfolio order combining multiple asset legs.
///
/// E.g. "long 0.20 BTC + short 0.15 ETH" → `ComboOrder` with 2 legs whose
/// fractions are sized jointly to satisfy portfolio constraints.
#[derive(Debug)]
pub struct ComboOrder {
    pub combo_id: String,
    pub generated_at_ms: u64,
    pub legs: Vec<OrderLeg>,
    /// Sum of per-leg expected log-growth (gross criterion before TC).
    pub total_expected_log_growth: f64,
    /// Sum of per-leg TC in bps: Σ turnover_i × tc_bps_i.
    pub total_tc_bps: f64,
    /// Sum of |position_fraction| across all legs.
    pub gross_exposure_fraction: f64,
    /// Signed net exposure: Σ (±position_fraction).
    pub net_exposure_fraction: f64,
}

impl ComboOrder {
    pub fn is_empty(&self) -> bool {
        self.legs.is_empty()
    }

    pub fn n_legs(&self) -> usize {
        self.legs.len()
    }

    /// Most common regime label across legs (modal label).
    pub fn regime_summary(&self) -> String {
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for leg in &self.legs {
            *counts.entry(leg.regime_label.as_str()).or_default() += 1;
        }
        counts
            .into_iter()
            .max_by_key(|(_, c)| *c)
            .map(|(r, _)| r.to_string())
            .unwrap_or_default()
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "combo_id": self.combo_id,
            "generated_at_ms": self.generated_at_ms,
            "n_legs": self.n_legs(),
            "total_expected_log_growth": self.total_expected_log_growth,
            "total_tc_bps": self.total_tc_bps,
            "gross_exposure_fraction": self.gross_exposure_fraction,
            "net_exposure_fraction": self.net_exposure_fraction,
            "regime": self.regime_summary(),
            "legs": self.legs.iter().map(|l| l.to_json()).collect::<Vec<_>>(),
        })
    }
}

// ── Per-asset decision (kept for backtest / DP paths) ─────────────────────────

/// Single-asset trade decision produced by `optimize()`.
///
/// Retained for backtest compatibility. New live code should use
/// `optimize_combo()` and `ComboOrder`.
#[derive(Debug, Clone)]
pub struct TradeDecision {
    pub asset: String,
    pub direction: Direction,
    pub position_fraction: f64,
    pub leverage: f64,
    pub position_size_usd: f64,
    pub expected_log_growth: f64,
    pub confidence: f64,
    pub regime_label: String,
}

impl TradeDecision {
    pub fn is_long(&self) -> bool {
        matches!(self.direction, Direction::Long)
    }
}

// ── Portfolio constraints ─────────────────────────────────────────────────────

/// Per-asset and portfolio-level optimization constraints.
///
/// All fields are fractions of NAV unless noted. Populated from env vars /
/// `RiskLimits` at runtime.
#[derive(Debug, Clone)]
pub struct PortfolioConstraints {
    pub max_position_fraction: f64,  // Single-asset max (e.g. 0.30)
    pub max_gross_exposure: f64,     // Σ|f_i| limit (e.g. 3.0)
    pub max_net_exposure: f64,       // |Σ f_i| limit (e.g. 1.8)
    pub max_leverage: f64,           // Per-asset leverage cap (e.g. 5.0)
    pub transaction_cost_bps: f64,   // One-way cost proxy in bps (e.g. 5.0)
    pub turnover_penalty: f64,       // λ_tc in the objective (e.g. 0.001)
    pub kelly_fraction: f64,         // Fractional Kelly multiplier (e.g. 0.3)
    pub min_confidence: f64,         // Minimum ensemble confidence to open
}

impl PortfolioConstraints {
    pub fn from_risk_config(max_lev: f64, max_weight: f64, kelly_frac: f64) -> Self {
        Self {
            max_position_fraction: max_weight,
            max_gross_exposure: max_lev,
            max_net_exposure: max_lev * 0.6,
            max_leverage: max_lev,
            transaction_cost_bps: 5.0,
            turnover_penalty: 0.001,
            kelly_fraction: kelly_frac,
            min_confidence: 0.55,
        }
    }
}

// ── Optimizer trait ───────────────────────────────────────────────────────────

/// Portfolio optimizer: maps per-asset distribution estimates + portfolio state
/// → a joint `ComboOrder`.
pub trait PortfolioOptimizer: Send + Sync {
    /// Joint multi-asset optimization (primary entry point).
    ///
    /// Selects the best subset of assets from `asset_dists`, sizes each leg via
    /// constrained Kelly, and enforces portfolio-level constraints jointly.
    /// Returns `None` if no trade passes filters.
    fn optimize_combo(
        &self,
        asset_dists: &[AssetDistribution],
        portfolio: &PortfolioState,
        constraints: &PortfolioConstraints,
        combo_config: &ComboConfig,
    ) -> Option<ComboOrder>;

    /// Single-period optimization for compatibility with backtest / DP paths.
    ///
    /// Default: wraps `optimize_combo` with stub `AssetDistribution`s built from
    /// raw `EnsembleDistribution`s (no entry price / CVaR data).
    fn optimize(
        &self,
        distributions: &HashMap<String, EnsembleDistribution>,
        portfolio: &PortfolioState,
        constraints: &PortfolioConstraints,
    ) -> Vec<TradeDecision> {
        let asset_dists: Vec<AssetDistribution> = distributions
            .iter()
            .map(|(asset, dist)| {
                let current_fraction = portfolio.positions.get(asset.as_str()).map(|p| {
                    let sign = if p.direction.is_long() { 1.0 } else { -1.0 };
                    sign * p.size_usd / portfolio.nav.max(1.0)
                }).unwrap_or(0.0);
                AssetDistribution {
                    asset: asset.clone(),
                    ensemble: dist.clone(),
                    current_fraction,
                    entry_price: 0.0,
                    estimated_tc_bps: constraints.transaction_cost_bps,
                    cvar_95: 0.0,
                    cvar_99: 0.0,
                }
            })
            .collect();

        let combo_config = ComboConfig::default();
        let combo = match self.optimize_combo(&asset_dists, portfolio, constraints, &combo_config) {
            Some(c) => c,
            None => return vec![],
        };

        combo.legs.into_iter().map(|leg| TradeDecision {
            asset: leg.asset,
            direction: leg.direction,
            position_fraction: leg.position_fraction,
            leverage: leg.leverage,
            position_size_usd: leg.position_size_usd,
            expected_log_growth: leg.expected_log_growth,
            confidence: leg.confidence,
            regime_label: leg.regime_label,
        }).collect()
    }

    /// Multi-period dynamic programming (default = single-period).
    fn optimize_dp(
        &self,
        distributions: &HashMap<String, EnsembleDistribution>,
        portfolio: &PortfolioState,
        constraints: &PortfolioConstraints,
        _horizon: usize,
    ) -> Vec<TradeDecision> {
        self.optimize(distributions, portfolio, constraints)
    }
}

pub mod kelly;
