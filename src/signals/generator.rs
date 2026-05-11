//! Signal generation: converts the joint portfolio optimization result into
//! structured trade signals ready for risk checking and execution.
//!
//! ## Pipeline
//! `generate_combo` takes a slice of per-asset `AssetDistribution`s built by
//! the main loop (DualKF → FrequencyExtractor → Hawkes → BayesianFusion) and
//! calls `KellyOptimizer::optimize_combo` to produce one `ComboOrder`.
//!
//! There are no batch-fitted statistical models in this path. All inputs are
//! derived online from the three probabilistic signal sources.

use crate::{
    config::Config,
    optimizer::{
        AssetDistribution, ComboConfig, ComboOrder, PortfolioConstraints,
        PortfolioOptimizer, kelly::KellyOptimizer,
    },
    risk::manager::PortfolioState,
};

// ── Direction ────────────────────────────────────────────────────────────────

/// Trade direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Direction {
    Long,
    Short,
}

impl Direction {
    pub fn is_long(&self) -> bool {
        matches!(self, Direction::Long)
    }
}

impl std::fmt::Display for Direction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Direction::Long => write!(f, "LONG"),
            Direction::Short => write!(f, "SHORT"),
        }
    }
}

// ── TradeSignal ───────────────────────────────────────────────────────────────

/// A fully specified single-asset trade signal used by `RiskManager` and
/// `Executor`. Produced by converting an `OrderLeg` via `leg_to_signal` in
/// the main loop.
#[derive(Debug, Clone)]
pub struct TradeSignal {
    pub signal_id: String,
    pub generated_at_ms: u64,
    pub asset: String,
    pub asset_index: usize,
    pub direction: Direction,
    pub entry_price: f64,
    pub stop_loss: f64,
    pub take_profit: f64,
    pub leverage: f64,
    pub position_size_usd: f64,
    pub expected_value: f64,
    pub signal_confidence: f64,
    pub directional_edge: f64,
    pub regime_label: String,
    pub ensemble_p_up: f64,
    pub ensemble_p_down: f64,
    pub ensemble_confidence: f64,
}

impl TradeSignal {
    pub fn risk_reward(&self) -> f64 {
        let gain = (self.take_profit - self.entry_price).abs();
        let loss = (self.stop_loss - self.entry_price).abs();
        if loss < 1e-10 { return 0.0; }
        gain / loss
    }

    pub fn stop_distance_pct(&self) -> f64 {
        if self.entry_price < 1e-10 { return 0.0; }
        (self.stop_loss - self.entry_price).abs() / self.entry_price
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "signal_id": self.signal_id,
            "generated_at_ms": self.generated_at_ms,
            "asset": self.asset,
            "asset_index": self.asset_index,
            "direction": self.direction.to_string(),
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "leverage": self.leverage,
            "position_size_usd": self.position_size_usd,
            "expected_value": self.expected_value,
            "signal_confidence": self.signal_confidence,
            "directional_edge": self.directional_edge,
            "risk_reward": self.risk_reward(),
            "regime": self.regime_label,
            "ensemble_p_up": self.ensemble_p_up,
            "ensemble_p_down": self.ensemble_p_down,
            "ensemble_confidence": self.ensemble_confidence,
        })
    }
}

// ── Combo generation ──────────────────────────────────────────────────────────

/// Generate a joint combo order from per-asset distribution estimates.
///
/// All inputs are online estimates — no batch-fitted models needed:
/// - `ensemble` from `BayesianFusion` over DualKF + FrequencyExtractor + Hawkes
/// - `entry_price` from the DualKF posterior level
/// - `estimated_tc_bps` from the live order book spread
pub fn generate_combo(
    asset_dists: &[AssetDistribution],
    portfolio: &PortfolioState,
    config: &Config,
) -> Option<ComboOrder> {
    let constraints = PortfolioConstraints::from_risk_config(
        config.max_portfolio_leverage,
        config.max_single_asset_weight,
        config.kelly_fraction,
    );
    let combo_config = ComboConfig::from_config(config);
    KellyOptimizer::new().optimize_combo(asset_dists, portfolio, &constraints, &combo_config)
}
