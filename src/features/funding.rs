//! Funding rate feature engineering.
//!
//! Perpetual funding rates serve as a synthetic options-like positioning signal.
//! Extreme funding z-scores indicate crowded positioning and are contrarian signals.

use std::collections::VecDeque;

use crate::data::store::FundingSample;

/// Derived funding rate features for one asset.
#[derive(Debug, Clone, Default)]
pub struct FundingFeatures {
    /// Current 8h funding rate.
    pub current_rate: f64,
    /// Annualised funding rate (× 3 × 365).
    pub annualised: f64,
    /// Z-score vs 30-day rolling mean/std of funding rates.
    pub z_score: f64,
    /// Funding momentum: current minus prior 8h period.
    pub momentum: f64,
    /// Spot–perp premium (mark - oracle) / oracle.
    pub premium_pct: f64,
}

impl FundingFeatures {
    pub fn compute(history: &VecDeque<FundingSample>, mark_px: f64, oracle_px: f64) -> Self {
        if history.is_empty() {
            return Self::default();
        }

        let current = history.back().unwrap();
        let rates: Vec<f64> = history.iter().map(|s| s.rate).collect();
        let n = rates.len() as f64;

        let mu = rates.iter().sum::<f64>() / n;
        let sigma = (rates.iter().map(|r| (r - mu).powi(2)).sum::<f64>() / n)
            .sqrt()
            .max(1e-15);

        let z_score = (current.rate - mu) / sigma;
        let momentum = if history.len() >= 2 {
            current.rate - history.iter().rev().nth(1).map(|s| s.rate).unwrap_or(0.0)
        } else {
            0.0
        };

        let premium_pct = if oracle_px > 1e-10 {
            (mark_px - oracle_px) / oracle_px
        } else {
            0.0
        };

        Self {
            current_rate: current.rate,
            annualised: current.rate * 3.0 * 365.0,
            z_score,
            momentum,
            premium_pct,
        }
    }

    /// Contrarian signal: positive when shorts are crowded, negative when longs are crowded.
    /// Scaled to [-1, 1] using a tanh squeeze.
    pub fn contrarian_signal(&self) -> f64 {
        (-self.z_score / 2.5).tanh()
    }

    /// Feature vector for ML models.
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.current_rate,
            self.annualised,
            self.z_score,
            self.momentum,
            self.premium_pct,
            self.contrarian_signal(),
        ]
    }
}

/// Volatility ratio between short-term and long-term realised vol.
/// VR > 1.5 indicates pre-move vol compression followed by expansion.
pub fn volatility_ratio(returns_1h: &[f64], returns_1d: &[f64]) -> f64 {
    let rv_short = annualised_vol(returns_1h, 24.0 * 365.0);
    let rv_long = annualised_vol(returns_1d, 365.0);
    if rv_long < 1e-10 {
        return 1.0;
    }
    rv_short / rv_long
}

fn annualised_vol(returns: &[f64], bars_per_year: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let mu = returns.iter().sum::<f64>() / returns.len() as f64;
    let var = returns.iter().map(|r| (r - mu).powi(2)).sum::<f64>() / returns.len() as f64;
    (var * bars_per_year).sqrt()
}
