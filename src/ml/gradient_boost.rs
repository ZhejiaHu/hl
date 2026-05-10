//! Gradient Boosted Trees — one model for directional classification
//! (ternary: up / flat / down) and one for forward volatility regression.
//!
//! Uses MSE residuals as pseudo-residuals for both tasks.  For the classifier
//! we train three one-vs-rest regressors and apply softmax at prediction time.

use anyhow::{bail, Result};

use super::tree::{Node, build_tree};

// ─── Regression model ───────────────────────────────────────────────────────

/// Gradient boosted regression model.
#[derive(Debug, Clone)]
pub struct GbRegressor {
    trees: Vec<Node>,
    learning_rate: f64,
    base_pred: f64,
    n_estimators: usize,
    max_depth: usize,
}

impl GbRegressor {
    pub fn new(n_estimators: usize, learning_rate: f64, max_depth: usize) -> Self {
        Self {
            trees: Vec::new(),
            learning_rate,
            base_pred: 0.0,
            n_estimators,
            max_depth,
        }
    }

    /// Fit the model.
    ///
    /// - `features`: n_samples × n_features matrix.
    /// - `targets`: n_samples continuous targets.
    pub fn fit(&mut self, features: &[Vec<f64>], targets: &[f64]) -> Result<()> {
        let n = features.len();
        if n == 0 || targets.len() != n {
            bail!("features and targets must have the same non-zero length");
        }
        if n < 20 {
            bail!("Need at least 20 samples to fit GBRegressor (have {})", n);
        }

        self.base_pred = targets.iter().sum::<f64>() / n as f64;
        let mut predictions = vec![self.base_pred; n];

        for _ in 0..self.n_estimators {
            let residuals: Vec<f64> =
                targets.iter().zip(&predictions).map(|(&y, &p)| y - p).collect();

            let tree = build_tree(features, &residuals, self.max_depth, 5);

            for (i, pred) in predictions.iter_mut().enumerate() {
                *pred += self.learning_rate * tree.predict(&features[i]);
            }
            self.trees.push(tree);
        }
        Ok(())
    }

    pub fn predict(&self, features: &[f64]) -> f64 {
        self.trees
            .iter()
            .fold(self.base_pred, |acc, t| acc + self.learning_rate * t.predict(features))
    }

    pub fn is_trained(&self) -> bool {
        !self.trees.is_empty()
    }
}

// ─── Ternary classifier (up / flat / down) ──────────────────────────────────

/// Gradient boosted ternary classifier via three one-vs-rest regressors.
///
/// Labels: -1 = down, 0 = flat, +1 = up.
#[derive(Debug, Clone)]
pub struct GbClassifier {
    /// One regressor per class (up, flat, down).
    regressors: [GbRegressor; 3],
    n_estimators: usize,
    learning_rate: f64,
    max_depth: usize,
}

impl GbClassifier {
    pub fn new(n_estimators: usize, learning_rate: f64, max_depth: usize) -> Self {
        Self {
            regressors: [
                GbRegressor::new(n_estimators, learning_rate, max_depth),
                GbRegressor::new(n_estimators, learning_rate, max_depth),
                GbRegressor::new(n_estimators, learning_rate, max_depth),
            ],
            n_estimators,
            learning_rate,
            max_depth,
        }
    }

    /// Fit.
    ///
    /// `labels`: integer slice where each entry is -1, 0, or +1.
    pub fn fit(&mut self, features: &[Vec<f64>], labels: &[i8]) -> Result<()> {
        let n = features.len();
        if n == 0 || labels.len() != n {
            bail!("features and labels must have the same non-zero length");
        }
        // One-hot encode each class.
        let up: Vec<f64> = labels.iter().map(|&l| if l == 1 { 1.0 } else { 0.0 }).collect();
        let flat: Vec<f64> = labels.iter().map(|&l| if l == 0 { 1.0 } else { 0.0 }).collect();
        let down: Vec<f64> = labels.iter().map(|&l| if l == -1 { 1.0 } else { 0.0 }).collect();

        self.regressors[0].fit(features, &up)?;
        self.regressors[1].fit(features, &flat)?;
        self.regressors[2].fit(features, &down)?;
        Ok(())
    }

    /// Returns (p_up, p_flat, p_down) after softmax.
    pub fn predict_proba(&self, features: &[f64]) -> (f64, f64, f64) {
        let s_up = self.regressors[0].predict(features);
        let s_flat = self.regressors[1].predict(features);
        let s_down = self.regressors[2].predict(features);
        softmax3(s_up, s_flat, s_down)
    }

    pub fn is_trained(&self) -> bool {
        self.regressors.iter().all(|r| r.is_trained())
    }
}

fn softmax3(a: f64, b: f64, c: f64) -> (f64, f64, f64) {
    let max = a.max(b).max(c);
    let ea = (a - max).exp();
    let eb = (b - max).exp();
    let ec = (c - max).exp();
    let s = ea + eb + ec;
    (ea / s, eb / s, ec / s)
}
