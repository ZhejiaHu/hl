use anyhow::{Context, Result};
use std::env;

#[derive(Debug, Clone)]
pub struct Config {
    // Exchange credentials
    pub private_key: String,
    pub wallet_address: String,
    pub use_testnet: bool,

    // Risk (delegated to RiskLimits at runtime, but stored here for backward compat)
    pub max_portfolio_leverage: f64,
    pub max_single_asset_weight: f64,
    pub daily_drawdown_limit: f64,
    pub target_daily_vol: f64,
    pub kelly_fraction: f64,

    // Historical bootstrap
    pub history_days: u64,

    // Execution
    pub signal_interval_secs: u64,
    pub max_signal_age_ms: u64,

    // Dual Kalman Filter
    pub dkf_sigma_obs: f64,
    pub dkf_sigma_param_walk: f64,

    // Frequency extractor
    pub freq_window_bars: usize,

    // Ensemble fusion
    pub fusion_evidence_decay: f64,
    pub fusion_min_weight: f64,

    // Backtest
    pub backtest_train_bars: usize,
    pub backtest_test_bars: usize,

    // Market impact model
    pub impact_eta: f64,
    pub impact_kappa: f64,

    // Joint combo optimizer
    pub max_combo_assets: usize,       // Cardinality: max legs in one combo
    pub max_combo_tc_bps: f64,         // Total TC budget across all legs in bps
    pub enumerate_combo_subsets: bool, // Exhaustive subset search when n ≤ 10
}

impl Config {
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            private_key: env::var("PRIVATE_KEY").context("PRIVATE_KEY not set")?,
            wallet_address: env::var("WALLET_ADDRESS").context("WALLET_ADDRESS not set")?,
            use_testnet: env::var("USE_TESTNET")
                .unwrap_or_default()
                .parse()
                .unwrap_or(false),

            max_portfolio_leverage: parse_env_f64("MAX_PORTFOLIO_LEVERAGE", 3.0)?,
            max_single_asset_weight: parse_env_f64("MAX_SINGLE_ASSET_WEIGHT", 0.30)?,
            daily_drawdown_limit: parse_env_f64("DAILY_DRAWDOWN_LIMIT", 0.08)?,
            target_daily_vol: parse_env_f64("TARGET_DAILY_VOL", 0.015)?,
            kelly_fraction: parse_env_f64("KELLY_FRACTION", 0.3)?,

            history_days: parse_env_u64("HISTORY_DAYS", 180)?,
            signal_interval_secs: parse_env_u64("SIGNAL_INTERVAL_SECS", 300)?,
            max_signal_age_ms: parse_env_u64("MAX_SIGNAL_AGE_MS", 5_000)?,

            dkf_sigma_obs: parse_env_f64("DKF_SIGMA_OBS", 1e-3)?,
            dkf_sigma_param_walk: parse_env_f64("DKF_SIGMA_PARAM_WALK", 1e-5)?,

            freq_window_bars: parse_env_usize("FREQ_WINDOW_BARS", 64)?,

            fusion_evidence_decay: parse_env_f64("FUSION_EVIDENCE_DECAY", 0.99)?,
            fusion_min_weight: parse_env_f64("FUSION_MIN_WEIGHT", 0.05)?,

            backtest_train_bars: parse_env_usize("BACKTEST_TRAIN_BARS", 500)?,
            backtest_test_bars: parse_env_usize("BACKTEST_TEST_BARS", 100)?,

            impact_eta: parse_env_f64("IMPACT_ETA", 0.1)?,
            impact_kappa: parse_env_f64("IMPACT_KAPPA", 0.3)?,

            max_combo_assets: parse_env_usize("MAX_COMBO_ASSETS", 3)?,
            max_combo_tc_bps: parse_env_f64("MAX_COMBO_TC_BPS", 25.0)?,
            enumerate_combo_subsets: env::var("ENUMERATE_COMBO_SUBSETS")
                .unwrap_or_default()
                .parse()
                .unwrap_or(true),
        })
    }
}

fn parse_env_f64(key: &str, default: f64) -> Result<f64> {
    Ok(env::var(key)
        .unwrap_or_else(|_| default.to_string())
        .parse()?)
}

fn parse_env_u64(key: &str, default: u64) -> Result<u64> {
    Ok(env::var(key)
        .unwrap_or_else(|_| default.to_string())
        .parse()?)
}

fn parse_env_usize(key: &str, default: usize) -> Result<usize> {
    Ok(env::var(key)
        .unwrap_or_else(|_| default.to_string())
        .parse()?)
}
