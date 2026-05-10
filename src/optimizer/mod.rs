//! Portfolio optimizer traits and shared types.
//!
//! The optimizer converts ensemble distributions into executable trade decisions
//! by solving a constrained expected-utility maximization problem.
//!
//! ## Optimization problem (per asset, single-period)
//!   maximize  E[log(1 + f · r)]  −  λ_tc · |f − f_prev|
//!   subject to:
//!     |f|           ≤ max_position_fraction
//!     Σ |f_i|       ≤ max_gross_exposure_fraction
//!     |Σ f_i · d_i| ≤ max_net_exposure_fraction   (d_i = ±1)
//!     f · leverage  ≤ max_leverage
//!
//! where f is the fractional position size (fraction of NAV).
//!
//! ## Dynamic programming hook
//! `PortfolioOptimizer::optimize_dp` provides an interface for multi-period DP.
//! The default implementation falls back to the single-period solution.

use std::collections::HashMap;

use crate::{
    ensemble::EnsembleDistribution,
    risk::manager::PortfolioState,
    signals::generator::Direction,
};

// ── Output types ──────────────────────────────────────────────────────────────

/// A fully computed trade decision produced by the optimizer.
#[derive(Debug, Clone)]
pub struct TradeDecision {
    pub asset: String,
    pub direction: Direction,
    /// Fractional position size as a share of NAV ∈ [0, max_position_fraction].
    pub position_fraction: f64,
    /// Leverage multiplier ≥ 1.
    pub leverage: f64,
    /// Absolute notional size in USD.
    pub position_size_usd: f64,
    /// Expected log-growth per bar under the optimizer's model.
    pub expected_log_growth: f64,
    /// Optimizer confidence (mirrors ensemble confidence).
    pub confidence: f64,
    /// Description of the distribution state ("trending", "high_vol", etc.).
    pub regime_label: String,
}

impl TradeDecision {
    pub fn is_long(&self) -> bool {
        matches!(self.direction, Direction::Long)
    }
}

// ── Constraints ───────────────────────────────────────────────────────────────

/// Per-asset and portfolio-level optimization constraints.
///
/// All fields are *fractions of NAV* unless noted. Loaded from config / risk
/// limits at runtime — no hardcoded values.
#[derive(Debug, Clone)]
pub struct PortfolioConstraints {
    pub max_position_fraction: f64,  // Single-asset max (e.g. 0.30)
    pub max_gross_exposure: f64,     // Sum of |f_i| (e.g. 3.0 for 3× gross)
    pub max_net_exposure: f64,       // |net long - net short| (e.g. 1.5)
    pub max_leverage: f64,           // Per-asset leverage cap (e.g. 5.0)
    pub transaction_cost_bps: f64,   // One-way cost in bps (e.g. 5.0)
    pub turnover_penalty: f64,       // λ_tc in the objective (e.g. 0.001)
    pub kelly_fraction: f64,         // Fractional Kelly multiplier (e.g. 0.3)
    pub min_confidence: f64,         // Minimum ensemble confidence to open (e.g. 0.55)
}

impl PortfolioConstraints {
    pub fn from_risk_config(
        max_lev: f64,
        max_weight: f64,
        kelly_frac: f64,
    ) -> Self {
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

/// Portfolio optimizer: maps distributions + portfolio state → trade decisions.
pub trait PortfolioOptimizer: Send + Sync {
    /// Single-period optimization (executed every tick).
    fn optimize(
        &self,
        distributions: &HashMap<String, EnsembleDistribution>,
        portfolio: &PortfolioState,
        constraints: &PortfolioConstraints,
    ) -> Vec<TradeDecision>;

    /// Multi-period dynamic programming (optional; default = single-period).
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
