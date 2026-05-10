//! Signal generation: converts the joint portfolio optimization result into
//! structured trade signals ready for risk checking and execution.
//!
//! ## Primary path (live trading)
//! `generate_combo` takes a slice of per-asset `AssetDistribution`s (built by
//! the main loop after estimation) and calls the `KellyOptimizer` to produce
//! one `ComboOrder` — a jointly optimized set of legs such as
//! "long 0.20 BTC + short 0.15 ETH".
//!
//! ## Backtest / compat path
//! `generate_signals` produces per-asset `TradeSignal`s from the older
//! `HashMap<String, EnsembleDistribution>` interface. Used by tests and legacy
//! single-asset simulation code.

use std::collections::HashMap;

use uuid::Uuid;

use crate::{
    assets::Asset,
    config::Config,
    ensemble::EnsembleDistribution,
    features::assembler::FeatureRow,
    optimizer::{
        AssetDistribution, ComboConfig, ComboOrder, PortfolioConstraints,
        PortfolioOptimizer, TradeDecision, kelly::KellyOptimizer,
    },
    risk::manager::PortfolioState,
    stats::distribution::ReturnDist,
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

// ── TradeSignal (backtest / single-asset compat) ──────────────────────────────

/// A fully specified single-asset trade signal ready for risk checking and
/// execution. Produced by `generate_signals` (backtest path) or by converting
/// an `OrderLeg` via `leg_to_signal` in the main loop.
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
    pub predicted_vol_4h: f64,
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
            "predicted_vol_4h": self.predicted_vol_4h,
        })
    }
}

// ── Primary path: joint combo generation ─────────────────────────────────────

/// Generate a joint combo order from per-asset distribution estimates.
///
/// This is the primary entry point for the live trading system. Each
/// `AssetDistribution` contains the fused ensemble, entry price, and transaction
/// cost estimate, so the optimizer can make fully-informed joint decisions
/// (e.g. "long BTC + short ETH") rather than sizing each asset independently.
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

// ── Backtest / compat path ────────────────────────────────────────────────────

/// Generate per-asset trade signals from ensemble distributions.
///
/// Retained for the backtest engine and any single-asset code paths.
/// Live trading code should use `generate_combo` instead.
pub fn generate_signals(
    assets: &[Asset],
    distributions: &HashMap<String, EnsembleDistribution>,
    feature_rows: &HashMap<String, FeatureRow>,
    portfolio: &PortfolioState,
    config: &Config,
    dist_models: &HashMap<String, ReturnDist>,
) -> Vec<TradeSignal> {
    let now_ms = timestamp_ms();
    let constraints = PortfolioConstraints::from_risk_config(
        config.max_portfolio_leverage,
        config.max_single_asset_weight,
        config.kelly_fraction,
    );

    let optimizer = KellyOptimizer::new();
    let decisions: Vec<TradeDecision> = optimizer.optimize(distributions, portfolio, &constraints);

    let mut signals: Vec<TradeSignal> = Vec::new();

    for decision in &decisions {
        let asset = match assets.iter().find(|a| a.symbol == decision.asset) {
            Some(a) => a,
            None => continue,
        };
        let dist = match distributions.get(&decision.asset) {
            Some(d) => d,
            None => continue,
        };
        let row = match feature_rows.get(&decision.asset) {
            Some(r) => r,
            None => continue,
        };

        let entry = row.kalman_level;
        if entry <= 0.0 { continue; }

        let cvar_95 = dist_models
            .get(&decision.asset)
            .map(|d| d.cvar(0.95).abs())
            .unwrap_or(0.0);
        let predicted_vol = dist.predictive_std.max(cvar_95).max(0.001);
        let atr_stop = 2.0 * predicted_vol * entry;
        let min_rr = regime_min_rr(dist);

        let (stop_loss, take_profit) = match decision.direction {
            Direction::Long => (entry - atr_stop, entry + atr_stop * min_rr),
            Direction::Short => (entry + atr_stop, entry - atr_stop * min_rr),
        };
        if stop_loss <= 0.0 || take_profit <= 0.0 { continue; }

        let ev = dist.directional_edge().abs() * dist.confidence;

        signals.push(TradeSignal {
            signal_id: Uuid::new_v4().to_string(),
            generated_at_ms: now_ms,
            asset: asset.symbol.clone(),
            asset_index: asset.index,
            direction: decision.direction,
            entry_price: entry,
            stop_loss,
            take_profit,
            leverage: decision.leverage,
            position_size_usd: decision.position_size_usd,
            expected_value: ev,
            signal_confidence: decision.confidence,
            directional_edge: dist.directional_edge(),
            predicted_vol_4h: predicted_vol,
            regime_label: decision.regime_label.clone(),
            ensemble_p_up: dist.p_up,
            ensemble_p_down: dist.p_down,
            ensemble_confidence: dist.confidence,
        });
    }

    signals.sort_by(|a, b| b.expected_value.partial_cmp(&a.expected_value).unwrap());
    let mut seen = std::collections::HashSet::new();
    signals
        .into_iter()
        .filter(|s| seen.insert(s.asset.clone()))
        .take(3)
        .collect()
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn regime_min_rr(dist: &EnsembleDistribution) -> f64 {
    match dist.regime_description() {
        "high_vol" => 3.0,
        "trending" => 2.0,
        "consolidating" => 1.5,
        _ => 2.0,
    }
}

fn timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
