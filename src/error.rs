use thiserror::Error;

#[derive(Error, Debug)]
pub enum TradingError {
    #[error("API error: {0}")]
    Api(#[from] anyhow::Error),

    #[error("Insufficient data: need at least {needed} bars, have {available}")]
    InsufficientData { needed: usize, available: usize },

    #[error("Model not trained")]
    ModelNotTrained,

    #[error("Risk check failed: {reason}")]
    RiskCheckFailed { reason: String },

    #[error("Execution error: {0}")]
    Execution(String),

    #[error("No signal generated")]
    NoSignal,
}
