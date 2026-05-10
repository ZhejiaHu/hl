//! State and parameter estimation traits.
//!
//! Every estimator in the system outputs a `StatePosterior` — a Gaussian summary
//! of the latent state — which is consumed by the ensemble fusion layer.
//!
//! Design: regime shifts emerge from the *evolution of the distribution* (changing
//! σ, κ, μ); there is no explicit hidden-state regime classifier.

use anyhow::Result;

// ── Core data types ───────────────────────────────────────────────────────────

/// A single price/volume observation fed to state estimators.
#[derive(Debug, Clone)]
pub struct Observation {
    pub price: f64,
    pub volume: f64,
    pub timestamp_ms: u64,
}

/// Posterior distribution over the latent state, produced by a `StateEstimator`.
///
/// The fields capture both point estimates and uncertainty, enabling downstream
/// components to propagate uncertainty rather than treating estimates as exact.
#[derive(Debug, Clone)]
pub struct StatePosterior {
    /// Estimated price level (Kalman-smoothed).
    pub level: f64,
    /// Estimated first-order trend (velocity per bar).
    pub velocity: f64,
    /// Estimated drift μ: unconditional per-bar expected return.
    pub drift: f64,
    /// Estimated mean-reversion speed κ ∈ [0, 1).
    pub kappa: f64,
    /// Estimated conditional state noise std σ_v.
    pub sigma: f64,
    /// Log one-step-ahead predictive density p(z_t | z_{1:t-1}).
    /// Used by `BayesianFusion` to weight estimators.
    pub log_predictive: f64,
    /// Posterior variance of the level estimate P_{t|t}[0,0].
    pub level_variance: f64,
}

impl StatePosterior {
    /// One-step-ahead predictive mean: level + velocity + drift.
    pub fn predictive_mean(&self) -> f64 {
        self.level + self.velocity + self.drift
    }

    /// One-step-ahead predictive std (combined state and observation uncertainty).
    pub fn predictive_std(&self) -> f64 {
        (self.level_variance + self.sigma * self.sigma).sqrt().max(1e-10)
    }

    /// Probability of a positive next-bar return under a Gaussian predictive model.
    pub fn p_up(&self) -> f64 {
        let z = self.predictive_mean() / self.predictive_std();
        standard_normal_cdf(z)
    }

    /// Probability of a negative next-bar return.
    pub fn p_down(&self) -> f64 {
        let z = -self.predictive_mean() / self.predictive_std();
        standard_normal_cdf(z)
    }
}

/// Estimated time-varying model parameters exposed by `ParameterEstimator`.
#[derive(Debug, Clone)]
pub struct EstimatedParameters {
    /// Per-bar drift (μ).
    pub drift: f64,
    /// Mean-reversion speed (κ).
    pub kappa: f64,
    /// State noise standard deviation (σ_v).
    pub vol_state: f64,
    /// Observation noise standard deviation (σ_w).
    pub vol_obs: f64,
}

// ── Estimator traits ──────────────────────────────────────────────────────────

/// Online state estimator: processes observations one-at-a-time and returns
/// an updated posterior.
pub trait StateEstimator: Send + Sync {
    /// Ingest a new observation and return the updated `StatePosterior`.
    fn update(&mut self, obs: &Observation) -> Result<StatePosterior>;

    /// Current posterior without updating (cheap read).
    fn posterior(&self) -> StatePosterior;

    /// Reset state (call after a data gap or instrument change).
    fn reset(&mut self);
}

/// Online parameter estimator: jointly tracks slowly-varying model parameters.
pub trait ParameterEstimator: Send + Sync {
    /// One EM update step using a mini-batch of observations.
    fn em_step(&mut self, obs: &[Observation]) -> Result<()>;

    /// Current parameter estimates.
    fn parameters(&self) -> EstimatedParameters;
}

// ── Gaussian helpers ──────────────────────────────────────────────────────────

/// Standard normal CDF via Abramowitz & Stegun approximation (max err ≈ 1.5e-7).
pub fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))
}

fn erf_approx(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741
                    + t * (-1.453_152_027 + t * 1.061_405_429))));
    let result = 1.0 - poly * (-x * x).exp();
    if x >= 0.0 { result } else { -result }
}

pub mod dual_kalman;
