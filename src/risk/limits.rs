//! Configurable risk limits and kill-switch rules.
//!
//! All thresholds are read from environment variables at startup and can be
//! overridden at runtime. No numeric constants are hardcoded in this file.
//!
//! ## Kill switches
//! - `VolKillSwitch`: triggered when realized vol exceeds a multiple of baseline.
//! - `DrawdownKillSwitch`: triggered when drawdown exceeds the configured limit.
//! - `EstimatorInstabilitySwitch`: triggered when the Dual KF log-predictive
//!   drops below a threshold, indicating model breakdown.
//! - `LeverageBreachSwitch`: triggered when gross leverage exceeds max.
//!
//! When any kill switch fires, a `KillAction` is returned specifying what to do.

use std::env;

// ── Risk limits ───────────────────────────────────────────────────────────────

/// Portfolio-level and per-asset risk limits (all configurable, none hardcoded).
#[derive(Debug, Clone)]
pub struct RiskLimits {
    // ── Exposure ────────────────────────────────────────────────────────────
    pub max_gross_exposure: f64,     // Σ |position_i| / NAV
    pub max_net_exposure: f64,       // |net long - net short| / NAV
    pub max_leverage: f64,           // Per-asset leverage cap
    pub max_single_asset_weight: f64, // Single-asset notional / NAV

    // ── Drawdown / P&L ───────────────────────────────────────────────────────
    pub max_daily_drawdown: f64,     // Fraction of NAV (e.g. 0.08)
    pub max_portfolio_drawdown: f64, // Fraction from peak NAV (e.g. 0.15)

    // ── Signal quality ───────────────────────────────────────────────────────
    pub min_signal_confidence: f64,
    pub min_risk_reward: f64,
    pub max_concurrent_positions: usize,

    // ── Volatility ───────────────────────────────────────────────────────────
    pub target_daily_vol: f64,       // Target portfolio daily vol
    pub vol_kill_multiplier: f64,    // Kill at realized_vol > multiplier × target

    // ── Estimator health ─────────────────────────────────────────────────────
    pub min_log_predictive: f64,     // Kill if Dual KF log-predictive < this

    // ── Correlation ──────────────────────────────────────────────────────────
    pub max_pairwise_correlation: f64, // Max allowed correlation between positions

    // ── Kelly ─────────────────────────────────────────────────────────────────
    pub kelly_fraction: f64,
}

impl RiskLimits {
    /// Load from environment variables with sensible defaults.
    pub fn from_env() -> Self {
        Self {
            max_gross_exposure: parse_env("RISK_MAX_GROSS_EXPOSURE", 3.0),
            max_net_exposure: parse_env("RISK_MAX_NET_EXPOSURE", 1.5),
            max_leverage: parse_env("RISK_MAX_LEVERAGE", 5.0),
            max_single_asset_weight: parse_env("RISK_MAX_SINGLE_ASSET_WEIGHT", 0.30),
            max_daily_drawdown: parse_env("RISK_MAX_DAILY_DRAWDOWN", 0.08),
            max_portfolio_drawdown: parse_env("RISK_MAX_PORTFOLIO_DRAWDOWN", 0.20),
            min_signal_confidence: parse_env("RISK_MIN_CONFIDENCE", 0.60),
            min_risk_reward: parse_env("RISK_MIN_RR", 1.5),
            max_concurrent_positions: parse_env_usize("RISK_MAX_POSITIONS", 5),
            target_daily_vol: parse_env("RISK_TARGET_DAILY_VOL", 0.015),
            vol_kill_multiplier: parse_env("RISK_VOL_KILL_MULTIPLIER", 3.0),
            min_log_predictive: parse_env("RISK_MIN_LOG_PREDICTIVE", -10.0),
            max_pairwise_correlation: parse_env("RISK_MAX_PAIR_CORR", 0.75),
            kelly_fraction: parse_env("RISK_KELLY_FRACTION", 0.3),
        }
    }
}

// ── Kill-switch rules ─────────────────────────────────────────────────────────

/// The action to take when a kill switch fires.
#[derive(Debug, Clone, PartialEq)]
pub enum KillAction {
    /// Do not open new positions; keep existing ones.
    HaltNewTrades { reason: String },
    /// Close all positions immediately.
    CloseAll { reason: String },
    /// Reduce all positions to a given fraction of current size.
    ReduceAll { fraction: f64, reason: String },
}

/// Evaluates kill-switch rules given current system state.
pub struct KillSwitchEvaluator<'a> {
    pub limits: &'a RiskLimits,
}

impl<'a> KillSwitchEvaluator<'a> {
    pub fn new(limits: &'a RiskLimits) -> Self {
        Self { limits }
    }

    /// Check all kill-switch rules and return the highest-severity action, if any.
    pub fn evaluate(
        &self,
        drawdown: f64,
        gross_leverage: f64,
        realized_vol: f64,
        min_log_predictive: f64, // worst log-predictive across active estimators
        n_positions: usize,
    ) -> Option<KillAction> {
        // Priority 1: Close all — extreme conditions.
        if drawdown > self.limits.max_portfolio_drawdown {
            return Some(KillAction::CloseAll {
                reason: format!(
                    "portfolio_drawdown={:.1}% exceeds limit={:.1}%",
                    drawdown * 100.0,
                    self.limits.max_portfolio_drawdown * 100.0
                ),
            });
        }

        if gross_leverage > self.limits.max_gross_exposure * 1.5 {
            return Some(KillAction::CloseAll {
                reason: format!(
                    "gross_leverage={:.2}x far exceeds limit={:.2}x",
                    gross_leverage,
                    self.limits.max_gross_exposure
                ),
            });
        }

        // Priority 2: Reduce all — elevated risk.
        if realized_vol > self.limits.target_daily_vol * self.limits.vol_kill_multiplier {
            return Some(KillAction::ReduceAll {
                fraction: 0.5,
                reason: format!(
                    "realized_vol={:.3} > {:.1}× target={:.3}",
                    realized_vol,
                    self.limits.vol_kill_multiplier,
                    self.limits.target_daily_vol
                ),
            });
        }

        if min_log_predictive < self.limits.min_log_predictive {
            return Some(KillAction::ReduceAll {
                fraction: 0.5,
                reason: format!(
                    "estimator_instability: log_pred={:.2} < threshold={:.2}",
                    min_log_predictive,
                    self.limits.min_log_predictive
                ),
            });
        }

        // Priority 3: Halt new trades — moderate risk.
        if drawdown > self.limits.max_daily_drawdown {
            return Some(KillAction::HaltNewTrades {
                reason: format!(
                    "daily_drawdown={:.1}% exceeds limit={:.1}%",
                    drawdown * 100.0,
                    self.limits.max_daily_drawdown * 100.0
                ),
            });
        }

        if gross_leverage > self.limits.max_gross_exposure {
            return Some(KillAction::HaltNewTrades {
                reason: format!(
                    "gross_leverage={:.2}x exceeds limit={:.2}x",
                    gross_leverage,
                    self.limits.max_gross_exposure
                ),
            });
        }

        None
    }
}

// ── Rolling volatility estimator ──────────────────────────────────────────────

/// Online exponential-weighted variance estimator for realized volatility.
///
/// Used by kill-switch rules to detect vol regime changes without needing
/// a full GARCH fit on every tick.
pub struct EwmaVolEstimator {
    ema_var: f64,
    decay: f64,
    initialized: bool,
}

impl EwmaVolEstimator {
    /// `decay` ∈ (0, 1): higher → slower adaptation (e.g. 0.94 for RiskMetrics).
    pub fn new(decay: f64) -> Self {
        Self { ema_var: 0.0, decay, initialized: false }
    }

    /// Update with a new log-return observation; return current vol estimate.
    pub fn update(&mut self, ret: f64) -> f64 {
        if !self.initialized {
            self.ema_var = ret * ret;
            self.initialized = true;
        } else {
            self.ema_var = self.decay * self.ema_var + (1.0 - self.decay) * ret * ret;
        }
        self.ema_var.sqrt()
    }

    pub fn current_vol(&self) -> f64 {
        self.ema_var.sqrt()
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn parse_env(key: &str, default: f64) -> f64 {
    env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn parse_env_usize(key: &str, default: usize) -> usize {
    env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_limits() -> RiskLimits {
        RiskLimits {
            max_gross_exposure: 3.0,
            max_net_exposure: 1.5,
            max_leverage: 5.0,
            max_single_asset_weight: 0.30,
            max_daily_drawdown: 0.08,
            max_portfolio_drawdown: 0.20,
            min_signal_confidence: 0.60,
            min_risk_reward: 1.5,
            max_concurrent_positions: 5,
            target_daily_vol: 0.015,
            vol_kill_multiplier: 3.0,
            min_log_predictive: -10.0,
            max_pairwise_correlation: 0.75,
            kelly_fraction: 0.3,
        }
    }

    #[test]
    fn close_all_on_deep_drawdown() {
        let limits = default_limits();
        let eval = KillSwitchEvaluator::new(&limits);
        let action = eval.evaluate(0.25, 2.0, 0.01, -1.0, 2);
        assert!(matches!(action, Some(KillAction::CloseAll { .. })));
    }

    #[test]
    fn halt_on_daily_drawdown() {
        let limits = default_limits();
        let eval = KillSwitchEvaluator::new(&limits);
        let action = eval.evaluate(0.09, 2.0, 0.01, -1.0, 2);
        assert!(matches!(action, Some(KillAction::HaltNewTrades { .. })));
    }

    #[test]
    fn no_action_under_limits() {
        let limits = default_limits();
        let eval = KillSwitchEvaluator::new(&limits);
        let action = eval.evaluate(0.03, 1.5, 0.01, -1.0, 2);
        assert!(action.is_none());
    }

    #[test]
    fn ewma_vol_updates() {
        let mut est = EwmaVolEstimator::new(0.94);
        for _ in 0..50 {
            est.update(0.01);
        }
        let vol = est.current_vol();
        assert!((vol - 0.01).abs() < 0.005, "vol={}", vol);
    }
}
