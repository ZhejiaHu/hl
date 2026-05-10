//! 2-D Kalman filter for latent trend extraction.
//!
//! State vector: x = [level, velocity]ᵀ
//! Transition:   x_t = F·x_{t-1} + w,  w ~ N(0, Q)
//! Observation:  z_t = H·x_t + v,       v ~ N(0, R)
//!
//! F = [[1, Δt], [0, 1]],  H = [1, 0],  Δt = 1 (per-bar).
//!
//! Process noise Q and observation noise R are held constant and can be tuned
//! or estimated via the EM algorithm offline.

use nalgebra::{Matrix2, Vector2};

/// Kalman filter state.
#[derive(Debug, Clone)]
pub struct KalmanFilter {
    /// State estimate [level, velocity].
    pub x: Vector2<f64>,
    /// State covariance.
    pub p: Matrix2<f64>,
    /// Transition matrix.
    f: Matrix2<f64>,
    /// Observation row vector h = [1, 0].
    h: Vector2<f64>,
    /// Process noise covariance.
    q: Matrix2<f64>,
    /// Observation noise variance.
    r: f64,
}

impl KalmanFilter {
    /// Create a new filter with given noise parameters.
    ///
    /// - `q_level`: process noise variance on level (e.g. 1e-4)
    /// - `q_velocity`: process noise variance on velocity (e.g. 1e-6)
    /// - `r`: observation noise variance (e.g. 1e-2)
    pub fn new(q_level: f64, q_velocity: f64, r: f64) -> Self {
        Self {
            x: Vector2::zeros(),
            p: Matrix2::identity() * 1.0,
            f: Matrix2::new(1.0, 1.0, 0.0, 1.0), // Δt = 1
            h: Vector2::new(1.0, 0.0),
            q: Matrix2::new(q_level, 0.0, 0.0, q_velocity),
            r,
        }
    }

    /// Default noise parameters suitable for crypto daily returns.
    pub fn default_crypto() -> Self {
        Self::new(1e-4, 1e-7, 1e-3)
    }

    /// Process a single new price observation and return the updated
    /// (level_estimate, velocity_estimate).
    pub fn update(&mut self, price: f64) -> (f64, f64) {
        // Predict
        let x_pred = self.f * self.x;
        let p_pred = self.f * self.p * self.f.transpose() + self.q;

        // Innovation (scalar): y = z - hᵀ x_pred
        let y = price - self.h.dot(&x_pred);
        // Innovation covariance (scalar): s = hᵀ P_pred h + r
        let s = self.h.dot(&(p_pred * self.h)) + self.r;

        // Kalman gain (column vector 2×1): k = P_pred h / s
        let k = p_pred * self.h / s;

        // Update state and covariance
        self.x = x_pred + k * y;
        self.p = (Matrix2::identity() - k * self.h.transpose()) * p_pred;

        (self.x[0], self.x[1])
    }

    /// Run the filter over a full price series, returning (levels, velocities).
    pub fn filter_series(&mut self, prices: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut levels = Vec::with_capacity(prices.len());
        let mut velocities = Vec::with_capacity(prices.len());
        for &p in prices {
            let (l, v) = self.update(p);
            levels.push(l);
            velocities.push(v);
        }
        (levels, velocities)
    }

    /// Latest level estimate.
    pub fn level(&self) -> f64 {
        self.x[0]
    }

    /// Latest velocity (trend) estimate.
    pub fn velocity(&self) -> f64 {
        self.x[1]
    }
}
