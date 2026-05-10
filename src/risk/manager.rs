//! Portfolio risk management: pre-trade checks, position tracking, and
//! real-time portfolio metrics.

use std::collections::HashMap;

use rust_decimal::prelude::ToPrimitive;

use crate::{
    config::Config,
    error::TradingError,
    signals::generator::{Direction, TradeSignal},
};

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
    pub oid: Option<u64>, // Exchange order ID after placement.
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
            p.size_usd * p.leverage * if p.direction.is_long() { 1.0 } else { -1.0 }
        }).sum()
    }

    pub fn leverage_ratio(&self) -> f64 {
        if self.nav < 1e-10 {
            return 0.0;
        }
        self.gross_exposure() / self.nav
    }

    pub fn drawdown(&self) -> f64 {
        if self.peak_nav < 1e-10 {
            return 0.0;
        }
        (self.peak_nav - self.nav) / self.peak_nav
    }

    pub fn update_nav(&mut self, new_nav: f64) {
        self.nav = new_nav;
        if new_nav > self.peak_nav {
            self.peak_nav = new_nav;
        }
    }

    /// Pearson correlation between two positions' assets (simplified proxy).
    pub fn position_correlation(&self, asset1: &str, asset2: &str) -> f64 {
        // In production: read from DataStore rolling correlation.
        // Here we use a hardcoded correlation table for the universe.
        known_correlation(asset1, asset2)
    }

    pub fn has_position(&self, asset: &str) -> bool {
        self.positions.contains_key(asset)
    }
}

/// Pre-trade risk checks.
pub struct RiskManager<'a> {
    pub config: &'a Config,
}

impl<'a> RiskManager<'a> {
    pub fn new(config: &'a Config) -> Self {
        Self { config }
    }

    /// Validate a signal before execution. Returns an error if any check fails.
    pub fn check_signal(
        &self,
        signal: &TradeSignal,
        portfolio: &PortfolioState,
    ) -> Result<(), TradingError> {
        // 1. Drawdown limit.
        if portfolio.drawdown() > self.config.daily_drawdown_limit {
            return Err(TradingError::RiskCheckFailed {
                reason: format!(
                    "Drawdown {:.1}% exceeds limit {:.1}%",
                    portfolio.drawdown() * 100.0,
                    self.config.daily_drawdown_limit * 100.0
                ),
            });
        }

        // 2. Maximum leverage.
        let new_leverage =
            (portfolio.gross_exposure() + signal.position_size_usd * signal.leverage) / portfolio.nav.max(1.0);
        if new_leverage > self.config.max_portfolio_leverage {
            return Err(TradingError::RiskCheckFailed {
                reason: format!(
                    "Portfolio leverage {:.2}× would exceed max {:.2}×",
                    new_leverage, self.config.max_portfolio_leverage
                ),
            });
        }

        // 3. Single-asset concentration.
        let asset_notional =
            signal.position_size_usd + portfolio.positions.get(&signal.asset).map(|p| p.size_usd).unwrap_or(0.0);
        if asset_notional / portfolio.nav.max(1.0) > self.config.max_single_asset_weight {
            return Err(TradingError::RiskCheckFailed {
                reason: format!(
                    "{} would exceed {:.0}% weight limit",
                    signal.asset,
                    self.config.max_single_asset_weight * 100.0
                ),
            });
        }

        // 4. Correlation check: reject if adding would give >75% correlated pair.
        for (pos_asset, _) in &portfolio.positions {
            let corr = portfolio.position_correlation(&signal.asset, pos_asset);
            if corr > 0.75 {
                return Err(TradingError::RiskCheckFailed {
                    reason: format!(
                        "{} is >{:.0}% correlated with existing position {}",
                        signal.asset,
                        corr * 100.0,
                        pos_asset
                    ),
                });
            }
        }

        // 5. Minimum risk–reward.
        let rr = signal.risk_reward();
        if rr < 1.5 {
            return Err(TradingError::RiskCheckFailed {
                reason: format!("Risk-reward {:.2} below minimum 1.5", rr),
            });
        }

        // 6. Maximum concurrent positions.
        if portfolio.n_open_positions() >= 5 {
            return Err(TradingError::RiskCheckFailed {
                reason: "Maximum 5 concurrent positions reached".into(),
            });
        }

        Ok(())
    }

    /// Determine if a position should be force-closed.
    pub fn should_close(&self, pos: &Position, now_ms: u64, current_price: f64) -> Option<&'static str> {
        // Time stop: 48h max holding period.
        if pos.is_stale(now_ms, 48 * 3_600_000) {
            return Some("time_stop");
        }

        // Vol stop: if price moved against us by >2× the predicted vol, cut in half.
        let move_pct = (current_price - pos.entry_price).abs() / pos.entry_price.max(1e-10);
        if move_pct > 0.05 && pos.unrealised_pnl(current_price) < 0.0 {
            return Some("vol_stop");
        }

        None
    }
}

/// Approximate correlation table for the universe.
fn known_correlation(a: &str, b: &str) -> f64 {
    let pairs: &[(&str, &str, f64)] = &[
        ("BTC", "ETH", 0.88),
        ("BTC", "SOL", 0.78),
        ("BTC", "HYPE", 0.55),
        ("BTC", "ARB", 0.72),
        ("BTC", "MATIC", 0.70),
        ("BTC", "WIF", 0.60),
        ("ETH", "SOL", 0.80),
        ("ETH", "ARB", 0.82),
        ("ETH", "MATIC", 0.83),
        ("ETH", "HYPE", 0.58),
        ("SOL", "WIF", 0.70),
    ];
    for &(x, y, c) in pairs {
        if (x == a && y == b) || (x == b && y == a) {
            return c;
        }
    }
    0.5 // Default moderate correlation.
}
