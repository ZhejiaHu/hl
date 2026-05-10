//! Signal generation: combines HMM regime state with ML ensemble output to
//! produce structured trade signals with entry, stop-loss, and take-profit.

use std::collections::HashMap;

use uuid::Uuid;

use crate::{
    assets::Asset,
    config::Config,
    features::assembler::FeatureRow,
    hmm::model::{Regime, RegimeState},
    ml::ensemble::EnsembleOutput,
    risk::manager::PortfolioState,
    stats::distribution::ReturnDist,
};

/// Trade direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// A fully specified trade signal ready for execution.
#[derive(Debug, Clone)]
pub struct TradeSignal {
    pub signal_id: String,
    pub generated_at_ms: u64,
    pub asset: String,
    pub asset_index: usize,
    pub direction: Direction,
    /// Kalman-filtered entry price (limit order target).
    pub entry_price: f64,
    pub stop_loss: f64,
    pub take_profit: f64,
    pub leverage: f64,
    pub position_size_usd: f64,
    /// Expected value (edge × confidence).
    pub expected_value: f64,
    pub signal_confidence: f64,
    pub directional_edge: f64,
    pub hmm_regime: Regime,
    pub hmm_probability: f64,
    pub predicted_vol_4h: f64,
}

impl TradeSignal {
    pub fn risk_reward(&self) -> f64 {
        let gain = (self.take_profit - self.entry_price).abs();
        let loss = (self.stop_loss - self.entry_price).abs();
        if loss < 1e-10 {
            return 0.0;
        }
        gain / loss
    }

    pub fn stop_distance_pct(&self) -> f64 {
        if self.entry_price < 1e-10 {
            return 0.0;
        }
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
            "hmm_regime": self.hmm_regime.label(),
            "hmm_probability": self.hmm_probability,
            "predicted_vol_4h": self.predicted_vol_4h,
        })
    }
}

/// Generate trade signals from regime + ML outputs.
///
/// `dist_models` optionally provides a per-asset fitted return distribution
/// (selected by `ReturnDist::fit_best`). When present, position sizing uses
/// the 95% CVaR as a tail-risk floor on effective volatility, so extremely
/// heavy-tailed assets receive smaller positions even if GARCH vol is low.
pub fn generate_signals(
    assets: &[Asset],
    outputs: &HashMap<String, EnsembleOutput>,
    feature_rows: &HashMap<String, FeatureRow>,
    regime: &RegimeState,
    portfolio: &PortfolioState,
    config: &Config,
    dist_models: &HashMap<String, ReturnDist>,
) -> Vec<TradeSignal> {
    // Abort in low-confidence regimes.
    if !regime.is_confident() {
        return vec![];
    }

    // No new signals in high-vol regime.
    if regime.regime == Regime::HighVol && portfolio.n_open_positions() > 0 {
        return vec![];
    }

    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);

    let mut candidates: Vec<(f64, TradeSignal)> = Vec::new();

    for asset in assets {
        let output = match outputs.get(&asset.symbol) {
            Some(o) => o,
            None => continue,
        };
        let row = match feature_rows.get(&asset.symbol) {
            Some(r) => r,
            None => continue,
        };

        // Minimum signal thresholds.
        if output.signal_confidence < 0.60 {
            continue;
        }
        if output.directional_edge.abs() < 0.25 {
            continue;
        }

        let direction = if output.directional_edge > 0.0 {
            Direction::Long
        } else {
            Direction::Short
        };

        // In consolidation, only allow mean-reversion (use OU z-score).
        if regime.regime == Regime::Consolidation {
            // Long only when spread is deeply negative (oversold vs. cointegration partner).
            if direction.is_long() && row.ou_z_score > -1.5 {
                continue;
            }
            if !direction.is_long() && row.ou_z_score < 1.5 {
                continue;
            }
        }

        let entry = row.kalman_level; // Kalman-filtered price as entry target.
        if entry <= 0.0 {
            continue;
        }

        // Vol-based stop distance.
        let atr_stop = 2.0 * output.predicted_vol_4h * entry;
        let (stop_loss, take_profit) = match direction {
            Direction::Long => (
                entry - atr_stop,
                entry + atr_stop * regime.regime.min_rr(),
            ),
            Direction::Short => (
                entry + atr_stop,
                entry - atr_stop * regime.regime.min_rr(),
            ),
        };

        if stop_loss <= 0.0 || take_profit <= 0.0 {
            continue;
        }

        // Fractional Kelly leverage.
        let p_win = if direction.is_long() { output.p_up } else { output.p_down };
        let p_loss = 1.0 - p_win;
        let avg_win = (take_profit - entry).abs() / entry;
        let avg_loss = (stop_loss - entry).abs() / entry;
        let kelly = if avg_win < 1e-10 || avg_loss < 1e-10 {
            0.0
        } else {
            (p_win * avg_win - p_loss * avg_loss) / avg_win
        };
        let leverage = (kelly * config.kelly_fraction)
            .clamp(1.0, regime.regime.max_leverage().min(asset.max_leverage as f64));

        // Position size from volatility targeting, with CVaR tail-risk floor.
        //
        // effective_vol = max(GARCH predicted_vol, |CVaR_95|)
        // This ensures assets with heavy tails (Cauchy / discrete) are sized
        // conservatively even when their conditional variance looks benign.
        let cvar_95 = dist_models
            .get(&asset.symbol)
            .map(|d| d.cvar(0.95).abs())
            .unwrap_or(0.0);
        let effective_vol = output.predicted_vol_4h.max(cvar_95).max(0.001);

        let nav = portfolio.nav;
        let target_vol = config.target_daily_vol;
        let position_size_usd = (nav * leverage * regime.regime.size_scale()
            * target_vol
            / effective_vol)
        .min(nav * config.max_single_asset_weight);

        let ev = output.directional_edge.abs() * output.signal_confidence;
        let score = output.score();

        let signal = TradeSignal {
            signal_id: Uuid::new_v4().to_string(),
            generated_at_ms: now_ms,
            asset: asset.symbol.clone(),
            asset_index: asset.index,
            direction,
            entry_price: entry,
            stop_loss,
            take_profit,
            leverage,
            position_size_usd,
            expected_value: ev,
            signal_confidence: output.signal_confidence,
            directional_edge: output.directional_edge,
            hmm_regime: regime.regime,
            hmm_probability: regime.probability,
            predicted_vol_4h: output.predicted_vol_4h,
        };

        candidates.push((score, signal));
    }

    // Rank by score, take top 3, deduplicate assets.
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let mut seen_assets = std::collections::HashSet::new();
    candidates
        .into_iter()
        .filter(|(_, s)| seen_assets.insert(s.asset.clone()))
        .take(3)
        .map(|(_, s)| s)
        .collect()
}
