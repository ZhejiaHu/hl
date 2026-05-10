//! Execution realism model.
//!
//! ## Components
//! 1. **Latency**: rejects signals older than `max_signal_age_ms`.
//! 2. **Market impact**: Almgren–Chriss linear model.
//!    - Permanent impact:  Δp_perm = η  · Q / ADV
//!    - Temporary impact:  Δp_temp = κ' · σ · √(Q / (ADV · T))
//! 3. **Fill probability**: logistic model on depth-to-size ratio.
//! 4. **Partial fills**: actual fill = order_size × fill_probability.
//!
//! ## Signal quality vs executable opportunity
//! The model distinguishes:
//! - **Signal quality**: ensemble confidence, predictive edge.
//! - **Executable opportunity**: spread cost + impact < expected edge.
//! Only signals where edge > total_cost are considered executable.

use crate::signals::generator::TradeSignal;

// ─── Impact parameters ────────────────────────────────────────────────────────

/// Almgren–Chriss market impact parameters (configurable, not hardcoded).
#[derive(Debug, Clone)]
pub struct ImpactParams {
    /// Permanent impact coefficient η (fraction of price per ADV fraction traded).
    pub eta: f64,
    /// Temporary impact coefficient κ' (Almgren-Chriss).
    pub kappa: f64,
    /// Fraction of 24h ADV used for impact scaling.
    pub adv_fraction: f64,
}

impl Default for ImpactParams {
    fn default() -> Self {
        // Calibrated to liquid crypto perpetuals on Hyperliquid.
        Self { eta: 0.1, kappa: 0.3, adv_fraction: 0.001 }
    }
}

impl ImpactParams {
    pub fn from_env() -> Self {
        use std::env;
        Self {
            eta: env::var("IMPACT_ETA").ok().and_then(|v| v.parse().ok()).unwrap_or(0.1),
            kappa: env::var("IMPACT_KAPPA").ok().and_then(|v| v.parse().ok()).unwrap_or(0.3),
            adv_fraction: env::var("IMPACT_ADV_FRAC").ok().and_then(|v| v.parse().ok()).unwrap_or(0.001),
        }
    }
}

// ─── Execution metrics ────────────────────────────────────────────────────────

/// Detailed execution metrics for a pending order.
#[derive(Debug, Clone)]
pub struct ExecutionMetrics {
    /// Adjusted entry price after spread cost.
    pub adjusted_entry_price: f64,
    /// Estimated permanent price impact (fraction of price).
    pub permanent_impact_pct: f64,
    /// Estimated temporary price impact (fraction of price).
    pub temporary_impact_pct: f64,
    /// Total cost as fraction of position: spread + permanent + temporary.
    pub total_cost_pct: f64,
    /// Estimated fill probability ∈ [0, 1].
    pub fill_probability: f64,
    /// Expected fill size after partial-fill adjustment.
    pub expected_fill_usd: f64,
    /// Whether the signal is still executable (edge > total cost).
    pub is_executable: bool,
    /// Reason for non-executability (empty if executable).
    pub rejection_reason: String,
}

impl ExecutionMetrics {
    fn not_executable(reason: &str) -> Self {
        Self {
            adjusted_entry_price: 0.0,
            permanent_impact_pct: 0.0,
            temporary_impact_pct: 0.0,
            total_cost_pct: 0.0,
            fill_probability: 0.0,
            expected_fill_usd: 0.0,
            is_executable: false,
            rejection_reason: reason.to_string(),
        }
    }
}

// ─── Execution model ──────────────────────────────────────────────────────────

/// Execution realism model.
pub struct ExecutionModel {
    /// Maximum age of a signal before it is stale (milliseconds).
    pub max_signal_age_ms: u64,
    /// Spread cost in basis points (one-way).
    pub spread_cost_bps: f64,
    /// Market impact parameters.
    pub impact: ImpactParams,
}

impl ExecutionModel {
    pub fn new(max_signal_age_ms: u64, spread_cost_bps: f64, impact: ImpactParams) -> Self {
        Self { max_signal_age_ms, spread_cost_bps, impact }
    }

    pub fn default_crypto() -> Self {
        Self::new(5_000, 5.0, ImpactParams::default())
    }

    pub fn from_env() -> Self {
        use std::env;
        let age = env::var("MAX_SIGNAL_AGE_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(5_000);
        let spread = env::var("SPREAD_COST_BPS").ok().and_then(|v| v.parse().ok()).unwrap_or(5.0);
        Self::new(age, spread, ImpactParams::from_env())
    }

    /// Evaluate whether a trade signal is executable given current market state.
    ///
    /// - `signal`: the proposed trade.
    /// - `now_ms`: current wall-clock time in milliseconds.
    /// - `adv_24h_usd`: 24-hour average daily volume in USD (for impact scaling).
    /// - `bid_ask_spread_pct`: current spread as fraction of mid price.
    /// - `book_depth_usd`: available depth within 1% of mid (for fill model).
    pub fn evaluate(
        &self,
        signal: &TradeSignal,
        now_ms: u64,
        adv_24h_usd: f64,
        bid_ask_spread_pct: f64,
        book_depth_usd: f64,
    ) -> ExecutionMetrics {
        // ── Latency check ────────────────────────────────────────────────────
        let signal_age = now_ms.saturating_sub(signal.generated_at_ms);
        if signal_age > self.max_signal_age_ms {
            return ExecutionMetrics::not_executable(&format!(
                "signal_stale: age={}ms > max={}ms",
                signal_age, self.max_signal_age_ms
            ));
        }

        let price = signal.entry_price;
        let size_usd = signal.position_size_usd * signal.leverage;

        // ── Spread cost ──────────────────────────────────────────────────────
        let spread_cost = bid_ask_spread_pct * 0.5; // half-spread for limit order

        // ── Permanent impact: Δp_perm = η · (size / ADV) ────────────────────
        let adv = adv_24h_usd.max(1.0);
        let size_fraction = size_usd / adv;
        let perm_impact = self.impact.eta * size_fraction;

        // ── Temporary impact: Δp_temp = κ' · σ · √(size / ADV) ─────────────
        // Use spread as a proxy for short-term volatility if not provided.
        let sigma_proxy = bid_ask_spread_pct.max(0.0001);
        let temp_impact = self.impact.kappa * sigma_proxy * size_fraction.sqrt();

        // Direction: long pays ask (positive impact), short pays bid (positive impact).
        let total_cost = spread_cost + perm_impact + temp_impact;

        // ── Fill probability: logistic on depth_to_size ratio ───────────────
        let depth_ratio = book_depth_usd / size_usd.max(1.0);
        let fill_prob = logistic(depth_ratio - 1.0); // P(fill) ≈ σ(depth/size - 1)

        // ── Expected fill size ───────────────────────────────────────────────
        let expected_fill = signal.position_size_usd * fill_prob;

        // ── Adjusted entry price ─────────────────────────────────────────────
        let price_adjustment = price * (spread_cost + perm_impact);
        let adjusted_entry = if signal.direction.is_long() {
            price + price_adjustment
        } else {
            price - price_adjustment
        };

        // ── Executability: net edge must exceed total cost ───────────────────
        let net_edge = signal.directional_edge.abs() - total_cost * 100.0; // convert to same scale
        let is_executable = net_edge > 0.0 && fill_prob > 0.3;
        let rejection_reason = if !is_executable {
            if fill_prob <= 0.3 {
                format!("low_fill_probability: {:.2}%", fill_prob * 100.0)
            } else {
                format!("edge_below_cost: edge={:.4} cost={:.4}", signal.directional_edge.abs(), total_cost)
            }
        } else {
            String::new()
        };

        ExecutionMetrics {
            adjusted_entry_price: adjusted_entry,
            permanent_impact_pct: perm_impact,
            temporary_impact_pct: temp_impact,
            total_cost_pct: total_cost,
            fill_probability: fill_prob,
            expected_fill_usd: expected_fill,
            is_executable,
            rejection_reason,
        }
    }
}

/// Logistic function σ(x) = 1 / (1 + e^{-x}).
fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signals::generator::{Direction, TradeSignal};

    fn make_signal(age_ms: u64, size: f64, edge: f64) -> TradeSignal {
        TradeSignal {
            signal_id: "test".into(),
            generated_at_ms: 1_000_000u64.saturating_sub(age_ms),
            asset: "BTC".into(),
            asset_index: 0,
            direction: Direction::Long,
            entry_price: 50_000.0,
            stop_loss: 49_000.0,
            take_profit: 52_000.0,
            leverage: 1.0,
            position_size_usd: size,
            expected_value: edge,
            signal_confidence: 0.7,
            directional_edge: edge,
            predicted_vol_4h: 0.02,
            regime_label: "trending".into(),
            ensemble_p_up: 0.6,
            ensemble_p_down: 0.2,
            ensemble_confidence: 0.7,
        }
    }

    #[test]
    fn stale_signal_rejected() {
        let model = ExecutionModel::default_crypto();
        let signal = make_signal(10_000, 1000.0, 0.5);
        let result = model.evaluate(&signal, 1_000_000, 1e8, 0.0001, 1e6);
        assert!(!result.is_executable);
        assert!(result.rejection_reason.contains("stale"));
    }

    #[test]
    fn fresh_signal_with_deep_book() {
        let model = ExecutionModel::default_crypto();
        let signal = make_signal(100, 1000.0, 0.5);
        let result = model.evaluate(&signal, 1_000_000, 1e8, 0.0001, 1e7);
        assert!(result.fill_probability > 0.7);
        assert!(result.total_cost_pct < 0.01);
    }
}
