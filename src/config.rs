use anyhow::{Context, Result};
use std::env;

#[derive(Debug, Clone)]
pub struct Config {
    pub private_key: String,
    pub wallet_address: String,
    pub use_testnet: bool,

    // Risk
    pub max_portfolio_leverage: f64,
    pub max_single_asset_weight: f64,
    pub daily_drawdown_limit: f64,
    pub target_daily_vol: f64,
    pub kelly_fraction: f64,

    // Model
    pub hmm_states: usize,
    pub gb_n_estimators: usize,
    pub gb_learning_rate: f64,
    pub gb_max_depth: usize,
    pub history_days: u64,

    // Execution
    pub signal_interval_secs: u64,
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
            max_portfolio_leverage: env::var("MAX_PORTFOLIO_LEVERAGE")
                .unwrap_or_else(|_| "3.0".into())
                .parse()?,
            max_single_asset_weight: env::var("MAX_SINGLE_ASSET_WEIGHT")
                .unwrap_or_else(|_| "0.30".into())
                .parse()?,
            daily_drawdown_limit: env::var("DAILY_DRAWDOWN_LIMIT")
                .unwrap_or_else(|_| "0.08".into())
                .parse()?,
            target_daily_vol: env::var("TARGET_DAILY_VOL")
                .unwrap_or_else(|_| "0.015".into())
                .parse()?,
            kelly_fraction: env::var("KELLY_FRACTION")
                .unwrap_or_else(|_| "0.3".into())
                .parse()?,
            hmm_states: env::var("HMM_STATES")
                .unwrap_or_else(|_| "4".into())
                .parse()?,
            gb_n_estimators: env::var("GB_N_ESTIMATORS")
                .unwrap_or_else(|_| "100".into())
                .parse()?,
            gb_learning_rate: env::var("GB_LEARNING_RATE")
                .unwrap_or_else(|_| "0.1".into())
                .parse()?,
            gb_max_depth: env::var("GB_MAX_DEPTH")
                .unwrap_or_else(|_| "3".into())
                .parse()?,
            history_days: env::var("HISTORY_DAYS")
                .unwrap_or_else(|_| "180".into())
                .parse()?,
            signal_interval_secs: env::var("SIGNAL_INTERVAL_SECS")
                .unwrap_or_else(|_| "300".into())
                .parse()?,
        })
    }
}
