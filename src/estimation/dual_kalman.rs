//! Dual Kalman Filter: joint online estimation of latent state and time-varying
//! model parameters (drift μ, mean-reversion κ, state noise σ_v).
//!
//! Framework: Wan & Nelson (2001) dual estimation via two coupled linear KFs.
//!
//! State model (discrete-time Ornstein–Uhlenbeck):
//!   level_{t+1}    = (1 − κ) · level_t + κ · θ_t + velocity_t + μ + w_l
//!   velocity_{t+1} = ρ_v · velocity_t + w_v
//!
//!   w_l ~ N(0, σ_v²),  w_v ~ N(0, (σ_v · λ)²)
//!
//! where θ_t is the EMA long-run mean, and κ ∈ [0, 1).
//!
//! Observation model:
//!   z_t = level_t + v_t,   v_t ~ N(0, σ_obs²)
//!
//! Parameter model (random walk in log/bounded space):
//!   μ_{t+1}       = μ_t + e_μ
//!   κ_{t+1}       = κ_t + e_κ
//!   log σ_v_{t+1} = log σ_v_t + e_σ
//!
//! The outer parameter KF uses the state KF's innovation as its "observation"
//! via a first-order sensitivity Jacobian (finite-difference approximation).

use anyhow::Result;
use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};

use super::{Observation, StatePosterior, StateEstimator};

// ─────────────────────────────────────────────────────────────────────────────

/// Velocity persistence factor (controls how quickly trend estimates decay).
const VEL_PERSISTENCE: f64 = 0.9;
/// Velocity noise scaling relative to level noise.
const VEL_NOISE_SCALE: f64 = 0.1;
/// Finite-difference step for parameter Jacobian.
const JAC_EPS: f64 = 1e-5;

// ─────────────────────────────────────────────────────────────────────────────

/// Dual Kalman Filter: simultaneous state tracking and parameter identification.
///
/// ## Mathematical assumptions
/// - State vector x = [level, velocity] follows a mean-reverting trend model.
/// - Parameters θ = [μ, κ, log σ_v] follow independent random walks.
/// - Observations are scalar prices corrupted by i.i.d. Gaussian noise.
/// - The regime at time t is fully characterised by the current distribution
///   (μ_t, κ_t, σ_v_t) — no discrete hidden-state classifier.
pub struct DualKalmanFilter {
    // ── Inner state KF ────────────────────────────────────────────────────────
    /// State estimate x = [level, velocity].
    x_state: Vector2<f64>,
    /// State covariance P.
    p_state: Matrix2<f64>,

    // ── Outer parameter KF ────────────────────────────────────────────────────
    /// Parameter estimate θ = [drift μ, kappa κ, log σ_v].
    x_param: Vector3<f64>,
    /// Parameter covariance Π.
    p_param: Matrix3<f64>,
    /// Parameter process noise (random-walk step variance per parameter).
    q_param: Matrix3<f64>,

    // ── Shared ────────────────────────────────────────────────────────────────
    /// Fixed observation noise std σ_obs.
    sigma_obs: f64,
    /// Exponential moving average long-run mean θ_t (the OU reversion target).
    rolling_mean: f64,
    /// EMA decay for rolling_mean (α ≈ 1/half-life in bars).
    mean_alpha: f64,

    // ── Diagnostics ───────────────────────────────────────────────────────────
    log_predictive: f64,
    initialized: bool,
}

impl DualKalmanFilter {
    /// Construct with explicit noise parameters.
    ///
    /// - `sigma_obs`: observation noise std (e.g. 0.001 for normalised price).
    /// - `sigma_param_walk`: std per step of each parameter random walk
    ///   (e.g. 1e-5 for slowly-varying parameters).
    pub fn new(sigma_obs: f64, sigma_param_walk: f64) -> Self {
        let qp = sigma_param_walk * sigma_param_walk;
        Self {
            x_state: Vector2::zeros(),
            p_state: Matrix2::identity(),
            // Initial: drift≈0, kappa≈0.05, log σ_v≈ln(0.001)
            x_param: Vector3::new(0.0, 0.05, (1e-3f64).ln()),
            p_param: Matrix3::identity() * 0.1,
            q_param: Matrix3::from_diagonal(&Vector3::new(qp, qp * 0.1, qp)),
            sigma_obs,
            rolling_mean: 0.0,
            mean_alpha: 0.01,
            log_predictive: 0.0,
            initialized: false,
        }
    }

    /// Default parameters tuned for 5-minute crypto bars.
    pub fn default_crypto() -> Self {
        Self::new(1e-3, 1e-5)
    }

    // ── Parameter accessors ───────────────────────────────────────────────────

    fn drift(&self) -> f64 {
        self.x_param[0].clamp(-0.01, 0.01)
    }

    fn kappa(&self) -> f64 {
        self.x_param[1].clamp(0.0, 0.98)
    }

    fn sigma_v(&self) -> f64 {
        self.x_param[2].clamp(-12.0, -1.0).exp()
    }

    // ── State KF helpers ──────────────────────────────────────────────────────

    fn state_transition(&self) -> Matrix2<f64> {
        let k = self.kappa();
        // level: mean-reverting to rolling_mean; velocity: persistent.
        Matrix2::new(1.0 - k, 1.0, 0.0, VEL_PERSISTENCE)
    }

    fn state_control_input(&self) -> Vector2<f64> {
        // Affine drift: κ·θ_t + μ added to level channel.
        Vector2::new(
            self.kappa() * self.rolling_mean + self.drift(),
            0.0,
        )
    }

    fn state_noise(&self) -> Matrix2<f64> {
        let sv2 = self.sigma_v().powi(2);
        Matrix2::new(sv2, 0.0, 0.0, sv2 * VEL_NOISE_SCALE)
    }

    /// Predicted state mean and covariance given current parameter estimates.
    fn predict_state(&self) -> (Vector2<f64>, Matrix2<f64>) {
        let f = self.state_transition();
        let q = self.state_noise();
        let x_pred = f * self.x_state + self.state_control_input();
        let p_pred = f * self.p_state * f.transpose() + q;
        (x_pred, p_pred)
    }

    /// Predicted observation (level component) from a given state.
    fn predicted_obs(x: &Vector2<f64>) -> f64 {
        x[0] // H = [1, 0]
    }
}

impl StateEstimator for DualKalmanFilter {
    /// Process one new price observation; returns updated `StatePosterior`.
    ///
    /// Algorithm per step:
    ///   1. Predict state using *current* parameter estimates.
    ///   2. Compute innovation = z − ĥ·x_pred.
    ///   3. Update state KF (inner).
    ///   4. Compute parameter Jacobian via finite difference.
    ///   5. Update parameter KF (outer) using same innovation.
    fn update(&mut self, obs: &Observation) -> Result<StatePosterior> {
        let z = obs.price;

        // Bootstrap: set initial state from first observation.
        if !self.initialized {
            self.x_state = Vector2::new(z, 0.0);
            self.rolling_mean = z;
            self.initialized = true;
            return Ok(self.posterior());
        }

        // Update rolling mean (EMA long-run reversion target).
        self.rolling_mean =
            (1.0 - self.mean_alpha) * self.rolling_mean + self.mean_alpha * z;

        // ── Parameter KF predict (random walk) ──────────────────────────────
        let x_param_pred = self.x_param;
        let p_param_pred = self.p_param + self.q_param;

        // ── State KF predict (using predicted parameters temporarily) ────────
        let saved = self.x_param;
        self.x_param = x_param_pred;
        let (x_pred, p_pred) = self.predict_state();
        self.x_param = saved;

        // ── Innovation ───────────────────────────────────────────────────────
        let z_pred = Self::predicted_obs(&x_pred);
        let innovation = z - z_pred;
        // Innovation variance: s = H·P_pred·Hᵀ + σ_obs²
        let s = p_pred[(0, 0)] + self.sigma_obs * self.sigma_obs;

        // ── State KF update ──────────────────────────────────────────────────
        // Kalman gain K = P_pred·Hᵀ / s  (2×1 vector, H=[1,0])
        let k_state = Vector2::new(p_pred[(0, 0)] / s, p_pred[(1, 0)] / s);
        let x_state_new = x_pred + k_state * innovation;
        let p_state_new = {
            let ikh = Matrix2::new(
                1.0 - k_state[0], -k_state[0],
                -k_state[1],       1.0 - k_state[1],
            );
            // Joseph form for numerical stability: (I-KH)P(I-KH)ᵀ + K·σ²_obs·Kᵀ
            ikh * p_pred * ikh.transpose()
                + Matrix2::new(
                    k_state[0] * k_state[0] * self.sigma_obs * self.sigma_obs,
                    k_state[0] * k_state[1] * self.sigma_obs * self.sigma_obs,
                    k_state[1] * k_state[0] * self.sigma_obs * self.sigma_obs,
                    k_state[1] * k_state[1] * self.sigma_obs * self.sigma_obs,
                )
        };

        // ── Log predictive density ───────────────────────────────────────────
        let log_pred = -0.5
            * (innovation * innovation / s + s.ln() + (2.0 * std::f64::consts::PI).ln());

        // ── Parameter KF: Jacobian dz_pred / dθ via finite differences ───────
        let h_param: Vector3<f64> = {
            let mut j = Vector3::zeros();
            for i in 0..3 {
                let mut p_hi = x_param_pred;
                let mut p_lo = x_param_pred;
                p_hi[i] += JAC_EPS;
                p_lo[i] -= JAC_EPS;

                self.x_param = p_hi;
                let (xp_hi, _) = self.predict_state();
                self.x_param = p_lo;
                let (xp_lo, _) = self.predict_state();

                j[i] = (Self::predicted_obs(&xp_hi) - Self::predicted_obs(&xp_lo))
                    / (2.0 * JAC_EPS);
            }
            self.x_param = saved;
            j
        };

        // Parameter innovation variance: H_p·Π_pred·H_pᵀ + s_obs
        let s_param = h_param.dot(&(p_param_pred * h_param)) + s;
        let k_param = p_param_pred * h_param / s_param.max(1e-30);
        let x_param_new = x_param_pred + k_param * innovation;
        let i3 = Matrix3::identity();
        let p_param_new = (i3 - k_param * h_param.transpose()) * p_param_pred;

        // ── Commit ───────────────────────────────────────────────────────────
        self.x_state = x_state_new;
        self.p_state = p_state_new;
        // Clamp parameters to physically meaningful ranges.
        self.x_param = Vector3::new(
            x_param_new[0].clamp(-0.01, 0.01),  // μ: bounded drift
            x_param_new[1].clamp(0.001, 0.98),  // κ: mean-reversion speed
            x_param_new[2].clamp(-12.0, -1.0),  // log σ_v
        );
        self.p_param = p_param_new;
        self.log_predictive = log_pred;

        Ok(self.posterior())
    }

    fn posterior(&self) -> StatePosterior {
        StatePosterior {
            level: self.x_state[0],
            velocity: self.x_state[1],
            drift: self.drift(),
            kappa: self.kappa(),
            sigma: self.sigma_v(),
            log_predictive: self.log_predictive,
            level_variance: self.p_state[(0, 0)].max(0.0),
        }
    }

    fn reset(&mut self) {
        self.initialized = false;
        self.x_state = Vector2::zeros();
        self.p_state = Matrix2::identity();
        self.log_predictive = 0.0;
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dual_kf_tracks_mean_reverting_series() {
        let mut kf = DualKalmanFilter::default_crypto();
        // Feed a mean-reverting series around 100.0.
        let mut rng_val = 100.0f64;
        for i in 0..200 {
            rng_val += 0.3 * (100.0 - rng_val) + 0.5 * ((i as f64 * 0.7).sin());
            let obs = Observation { price: rng_val, volume: 1.0, timestamp_ms: i * 5000 };
            kf.update(&obs).unwrap();
        }
        let p = kf.posterior();
        // Level estimate should be close to the long-run mean.
        assert!((p.level - 100.0).abs() < 5.0, "level={}", p.level);
        // Kappa should have converged to a positive value.
        assert!(p.kappa > 0.0, "kappa={}", p.kappa);
    }

    #[test]
    fn dual_kf_detects_trending_series() {
        let mut kf = DualKalmanFilter::default_crypto();
        for i in 0..200 {
            let price = 100.0 + 0.05 * i as f64;
            let obs = Observation { price, volume: 1.0, timestamp_ms: i * 5000 };
            kf.update(&obs).unwrap();
        }
        let p = kf.posterior();
        // Velocity should be positive for a trending series.
        assert!(p.velocity > 0.0, "velocity={}", p.velocity);
    }
}
