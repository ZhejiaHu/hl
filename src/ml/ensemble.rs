//! Two-model ensemble: directional classifier + volatility regressor.
//!
//! Training targets:
//!  - Classifier: sign of forward 4h log-return, thresholded at ±0.3%
//!  - Regressor:  forward 4h realised volatility (std of 5m returns in that window)

use std::collections::HashMap;

use anyhow::{bail, Result};

use crate::features::assembler::FeatureRow;

use super::gradient_boost::{GbClassifier, GbRegressor};

/// Output of the ensemble inference step.
#[derive(Debug, Clone)]
pub struct EnsembleOutput {
    pub asset: String,
    pub p_up: f64,
    pub p_flat: f64,
    pub p_down: f64,
    /// Directional edge: P_up - P_down ∈ [-1, 1].
    pub directional_edge: f64,
    /// Signal confidence based on distributional entropy.
    pub signal_confidence: f64,
    /// Predicted 4h forward volatility.
    pub predicted_vol_4h: f64,
}

impl EnsembleOutput {
    /// Should we consider a LONG trade?
    pub fn is_long_signal(&self) -> bool {
        self.directional_edge > 0.25 && self.signal_confidence > 0.60
    }

    /// Should we consider a SHORT trade?
    pub fn is_short_signal(&self) -> bool {
        self.directional_edge < -0.25 && self.signal_confidence > 0.60
    }

    /// Score used for ranking across assets (higher = more attractive trade).
    pub fn score(&self) -> f64 {
        self.directional_edge.abs() * self.signal_confidence / self.predicted_vol_4h.max(0.001)
    }
}

/// Ensemble of directional classifier + volatility regressor, one per asset.
pub struct ModelEnsemble {
    classifiers: HashMap<String, GbClassifier>,
    regressors: HashMap<String, GbRegressor>,
    n_estimators: usize,
    learning_rate: f64,
    max_depth: usize,
}

impl ModelEnsemble {
    pub fn new(n_estimators: usize, learning_rate: f64, max_depth: usize) -> Self {
        Self {
            classifiers: HashMap::new(),
            regressors: HashMap::new(),
            n_estimators,
            learning_rate,
            max_depth,
        }
    }

    /// Train both models for `asset` using historical feature rows.
    ///
    /// `rows` must be in chronological order.  We need at least `horizon + 10`
    /// rows to generate training labels.
    ///
    /// `horizon`: number of 5m bars ahead used to compute forward return/vol (48 ≈ 4h).
    pub fn train(&mut self, asset: &str, rows: &[FeatureRow], horizon: usize) -> Result<()> {
        if rows.len() < horizon + 20 {
            bail!(
                "Need ≥ {} rows to train {}; have {}",
                horizon + 20,
                asset,
                rows.len()
            );
        }

        let n_train = rows.len() - horizon;
        let features: Vec<Vec<f64>> = rows[..n_train].iter().map(|r| r.to_feature_vec()).collect();

        // Classification labels: forward 4h return > +0.3% → +1, < -0.3% → -1, else 0.
        let labels: Vec<i8> = (0..n_train)
            .map(|i| {
                let fwd_close = rows[i + horizon].kalman_level; // Kalman-smoothed future price
                let cur_close = rows[i].kalman_level;
                if cur_close < 1e-10 {
                    return 0i8;
                }
                let ret = (fwd_close / cur_close).ln();
                if ret > 0.003 {
                    1
                } else if ret < -0.003 {
                    -1
                } else {
                    0
                }
            })
            .collect();

        // Regression targets: forward 4h realised vol.
        // Approximate as std of the `horizon` subsequent log-returns scaled to 4h.
        let vol_targets: Vec<f64> = (0..n_train)
            .map(|i| {
                let end = (i + horizon).min(rows.len() - 1);
                // Use Kalman velocity as proxy for instantaneous vol.
                let velocities: Vec<f64> = rows[i..end]
                    .iter()
                    .map(|r| r.kalman_velocity.abs())
                    .collect();
                if velocities.is_empty() {
                    return 0.01;
                }
                let mean = velocities.iter().sum::<f64>() / velocities.len() as f64;
                let std = (velocities.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                    / velocities.len() as f64)
                    .sqrt();
                (std * (horizon as f64).sqrt()).max(0.001)
            })
            .collect();

        let mut classifier = GbClassifier::new(self.n_estimators, self.learning_rate, self.max_depth);
        classifier.fit(&features, &labels)?;
        self.classifiers.insert(asset.to_string(), classifier);

        let mut regressor = GbRegressor::new(self.n_estimators, self.learning_rate, self.max_depth);
        regressor.fit(&features, &vol_targets)?;
        self.regressors.insert(asset.to_string(), regressor);

        Ok(())
    }

    /// Run inference for `asset` on a single feature row.
    pub fn predict(&self, asset: &str, row: &FeatureRow) -> Option<EnsembleOutput> {
        let clf = self.classifiers.get(asset)?;
        let reg = self.regressors.get(asset)?;
        if !clf.is_trained() || !reg.is_trained() {
            return None;
        }

        let fv = row.to_feature_vec();
        let (p_up, p_flat, p_down) = clf.predict_proba(&fv);
        let predicted_vol_4h = reg.predict(&fv).max(0.001);

        let directional_edge = p_up - p_down;

        // Entropy-based confidence: 1 - H / H_max
        let eps = 1e-10;
        let entropy = -(p_up * (p_up + eps).ln()
            + p_flat * (p_flat + eps).ln()
            + p_down * (p_down + eps).ln());
        let max_entropy = (3.0f64).ln();
        let signal_confidence = (1.0 - entropy / max_entropy).max(0.0);

        Some(EnsembleOutput {
            asset: asset.to_string(),
            p_up,
            p_flat,
            p_down,
            directional_edge,
            signal_confidence,
            predicted_vol_4h,
        })
    }

    pub fn is_trained(&self, asset: &str) -> bool {
        self.classifiers.get(asset).is_some_and(|c| c.is_trained())
            && self.regressors.get(asset).is_some_and(|r| r.is_trained())
    }
}
