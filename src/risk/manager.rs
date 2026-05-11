//! Portfolio risk management: pre-trade checks, position tracking, and
//! real-time portfolio metrics.
//!
//! All thresholds are read from `RiskLimits` (configurable, data-driven).
//! Correlation estimates are provided externally rather than from a hardcoded
//! table, enabling data-driven pair-risk assessment.

use std::collections::HashMap;

use rust_decimal::prelude::ToPrimitive;

use crate::{
    error::TradingError,
    signals::generator::{Direction, TradeSignal},
};

use super::limits::RiskLimits;

// ── Position ──────────────────────────────────────────────────────────────────

/// A single open position.
#[derive(Debug, Clone)]
pub struct Position {
    pub asset: String,
    pub direction: Direction,
    pub entry_price: f64,
    pub size_usd: f64,
    pub leverage: f64,
    pub stop_loss: f64,
    pub take_profit: f64,
    pub open_time_ms: u64,
    pub oid: Option<u64>,
    pub sl_oid: Option<u64>,
    pub tp_oid: Option<u64>,
}

impl Position {
    pub fn unrealised_pnl(&self, current_price: f64) -> f64 {
        if self.entry_price < 1e-10 {
            return 0.0;
        }
        let ret = match self.direction {
            Direction::Long => (current_price - self.entry_price) / self.entry_price,
            Direction::Short => (self.entry_price - current_price) / self.entry_price,
        };
        self.size_usd * ret
    }

    pub fn is_stale(&self, now_ms: u64, max_duration_ms: u64) -> bool {
        now_ms.saturating_sub(self.open_time_ms) > max_duration_ms
    }
}

// ── Portfolio state ───────────────────────────────────────────────────────────

/// Live portfolio state.
#[derive(Debug, Clone)]
pub struct PortfolioState {
    pub nav: f64,
    pub peak_nav: f64,
    pub positions: HashMap<String, Position>,
    pub total_realised_pnl: f64,
}

impl PortfolioState {
    pub fn new(initial_nav: f64) -> Self {
        Self {
            nav: initial_nav,
            peak_nav: initial_nav,
            positions: HashMap::new(),
            total_realised_pnl: 0.0,
        }
    }

    pub fn n_open_positions(&self) -> usize {
        self.positions.len()
    }

    pub fn gross_exposure(&self) -> f64 {
        self.positions.values().map(|p| p.size_usd * p.leverage).sum()
    }

    pub fn net_exposure(&self) -> f64 {
        self.positions.values().map(|p| {
            let sign = if p.direction.is_long() { 1.0 } else { -1.0 };
            p.size_usd * p.leverage * sign
        }).sum()
    }

    pub fn leverage_ratio(&self) -> f64 {
        if self.nav < 1e-10 { return 0.0; }
        self.gross_exposure() / self.nav
    }

    pub fn drawdown(&self) -> f64 {
        if self.peak_nav < 1e-10 { return 0.0; }
        (self.peak_nav - self.nav) / self.peak_nav
    }

    pub fn update_nav(&mut self, new_nav: f64) {
        self.nav = new_nav;
        if new_nav > self.peak_nav {
            self.peak_nav = new_nav;
        }
    }

    /// Return correlation between two assets.
    ///
    /// `corr_matrix`: optional externally-supplied rolling correlation data
    /// (keyed by sorted asset pair "A/B"). Falls back to a conservative 0.5.
    pub fn position_correlation(
        &self,
        asset1: &str,
        asset2: &str,
        corr_matrix: Option<&HashMap<String, f64>>,
    ) -> f64 {
        if let Some(cm) = corr_matrix {
            let key = if asset1 < asset2 {
                format!("{}/{}", asset1, asset2)
            } else {
                format!("{}/{}", asset2, asset1)
            };
            if let Some(&c) = cm.get(&key) {
                return c;
            }
        }
        0.5 // Conservative default; no hardcoded asset-specific values.
    }

    pub fn has_position(&self, asset: &str) -> bool {
        self.positions.contains_key(asset)
    }
}

// ── Risk manager ──────────────────────────────────────────────────────────────

/// Pre-trade risk checks driven entirely by `RiskLimits`.
pub struct RiskManager<'a> {
    pub limits: &'a RiskLimits,
}

impl<'a> RiskManager<'a> {
    pub fn new(limits: &'a RiskLimits) -> Self {
        Self { limits }
    }

    /// Validate a signal before execution.
    ///
    /// `corr_matrix`: rolling correlation data for pair-concentration check.
    pub fn check_signal(
        &self,
        signal: &TradeSignal,
        portfolio: &PortfolioState,
        corr_matrix: Option<&HashMap<String, f64>>,
    ) -> Result<(), TradingError> {
        // 1. Portfolio drawdown limit.
        if portfolio.drawdown() > self.limits.max_daily_drawdown {
            return Err(TradingError::RiskCheckFailed {
                reason: format!(
                    "Drawdown {:.1}% exceeds limit {:.1}%",
                    portfolio.drawdown() * 100.0,
                    self.limits.max_daily_drawdown * 100.0
                ),
            });
        }

        // 2. Portfolio leverage limit.
        let new_leverage = (portfolio.gross_exposure()
            + signal.position_size_usd * signal.leverage)
            / portfolio.nav.max(1.0);
        if new_leverage > self.limits.max_gross_exposure {
            return Err(TradingError::RiskCheckFailed {
                reason: format!(
                    "Portfolio leverage {:.2}x would exceed limit {:.2}x",
                    new_leverage, self.limits.max_gross_exposure
                ),
            });
        }

        // 3. Single-asset concentration.
        let asset_notional = signal.position_size_usd
            + portfolio.positions.get(&signal.asset).map(|p| p.size_usd).unwrap_or(0.0);
        if asset_notional / portfolio.nav.max(1.0) > self.limits.max_single_asset_weight {
            return Err(TradingError::RiskCheckFailed {
                reason: format!(
                    "{} would exceed {:.0}% weight limit",
                    signal.asset,
                    self.limits.max_single_asset_weight * 100.0
                ),
            });
        }

        // 4. Correlation check (data-driven via rolling correlation matrix).
        for pos_asset in portfolio.positions.keys() {
            let corr = portfolio.position_correlation(&signal.asset, pos_asset, corr_matrix);
            if corr > self.limits.max_pairwise_correlation {
                return Err(TradingError::RiskCheckFailed {
                    reason: format!(
                        "{} correlation {:.0}% with {} exceeds limit {:.0}%",
                        signal.asset,
                        corr * 100.0,
                        pos_asset,
                        self.limits.max_pairwise_correlation * 100.0
                    ),
                });
            }
        }

        // 5. Minimum risk-reward.
        let rr = signal.risk_reward();
        if rr < self.limits.min_risk_reward {
            return Err(TradingError::RiskCheckFailed {
                reason: format!("Risk-reward {:.2} below minimum {:.2}", rr, self.limits.min_risk_reward),
            });
        }

        // 6. Maximum concurrent positions.
        if portfolio.n_open_positions() >= self.limits.max_concurrent_positions {
            return Err(TradingError::RiskCheckFailed {
                reason: format!(
                    "Maximum {} concurrent positions reached",
                    self.limits.max_concurrent_positions
                ),
            });
        }

        // 7. Signal confidence.
        if signal.signal_confidence < self.limits.min_signal_confidence {
            return Err(TradingError::RiskCheckFailed {
                reason: format!(
                    "Signal confidence {:.2} below minimum {:.2}",
                    signal.signal_confidence, self.limits.min_signal_confidence
                ),
            });
        }

        Ok(())
    }

    /// Determine if a position should be force-closed.
    ///
    /// `max_hold_ms`: maximum holding period in milliseconds (from limits or
    /// strategy config). Set to 48 × 3_600_000 for 48h.
    pub fn should_close(
        &self,
        pos: &Position,
        now_ms: u64,
        current_price: f64,
        max_hold_ms: u64,
    ) -> Option<&'static str> {
        // Time stop.
        if pos.is_stale(now_ms, max_hold_ms) {
            return Some("time_stop");
        }

        // Volatility stop: price moved >5% against us.
        let move_pct = (current_price - pos.entry_price).abs() / pos.entry_price.max(1e-10);
        if move_pct > 0.05 && pos.unrealised_pnl(current_price) < 0.0 {
            return Some("vol_stop");
        }

        None
    }
}
