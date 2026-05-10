//! Probabilistic signal ensemble layer.
//!
//! ## Architecture
//! Each signal source (`ProbabilisticSignal`) outputs a `SignalOutput` containing
//! its predictive distribution and log-marginal-likelihood estimate.
//! `BayesianFusion` combines signals weighted by their recent predictive accuracy
//! and returns an `EnsembleDistribution` — the unified regime representation.
//!
//! ## No discrete regime labels
//! The distribution *is* the regime:
//! - Low σ, high |μ|, low κ → trending.
//! - Low σ, low |μ|, high κ → consolidating / mean-reverting.
//! - High σ → high-volatility chaos.
//! Downstream components adapt naturally to the distribution without labels.

/// Output of one probabilistic signal source.
#[derive(Debug, Clone)]
pub struct SignalOutput {
    /// Signal source identifier (for logging and weight tracking).
    pub source: &'static str,
    /// Probability of a positive next-bar return.
    pub p_up: f64,
    /// Probability of a negative next-bar return.
    pub p_down: f64,
    /// Predictive distribution mean.
    pub predictive_mean: f64,
    /// Predictive distribution std.
    pub predictive_std: f64,
    /// Log marginal likelihood of the most recent observation under this model.
    /// Used by `BayesianFusion` to update model weights.
    pub log_evidence: f64,
}

impl SignalOutput {
    /// Probability of a flat (near-zero) return.
    pub fn p_flat(&self) -> f64 {
        (1.0 - self.p_up - self.p_down).clamp(0.0, 1.0)
    }

    /// Directional edge ∈ [−1, 1]: positive → bullish, negative → bearish.
    pub fn directional_edge(&self) -> f64 {
        self.p_up - self.p_down
    }

    /// Entropy-based confidence: 1 − H / H_max.
    pub fn confidence(&self) -> f64 {
        let pu = self.p_up.max(1e-10);
        let pd = self.p_down.max(1e-10);
        let pf = self.p_flat().max(1e-10);
        let h = -(pu * pu.ln() + pd * pd.ln() + pf * pf.ln());
        let h_max = (3.0f64).ln();
        (1.0 - h / h_max).clamp(0.0, 1.0)
    }
}

/// Fused ensemble distribution: the primary output of the estimation layer.
///
/// This struct replaces the old `HMM RegimeState`.  The distribution parameters
/// carry all regime information needed by the optimizer.
#[derive(Debug, Clone)]
pub struct EnsembleDistribution {
    /// Bayesian model-averaged probability of up move.
    pub p_up: f64,
    /// Bayesian model-averaged probability of down move.
    pub p_down: f64,
    /// Mixture predictive mean (sum of weight·mean_i).
    pub predictive_mean: f64,
    /// Mixture predictive std (accounts for within- and between-model variance).
    pub predictive_std: f64,
    /// Weighted-average signal confidence.
    pub confidence: f64,
    /// Per-source weights used in this fusion step (for auditability).
    pub weights: Vec<(&'static str, f64)>,
}

impl EnsembleDistribution {
    /// Directional edge: p_up − p_down ∈ [−1, 1].
    pub fn directional_edge(&self) -> f64 {
        self.p_up - self.p_down
    }

    /// True when the ensemble is confident enough to trade.
    pub fn is_confident(&self) -> bool {
        self.confidence > 0.55
    }

    /// Variance implied by the predictive std.
    pub fn predictive_variance(&self) -> f64 {
        self.predictive_std * self.predictive_std
    }

    /// Regime description derived purely from the distribution (no label mapping).
    pub fn regime_description(&self) -> &'static str {
        let edge = self.directional_edge().abs();
        let vol = self.predictive_std;
        match (edge, vol) {
            _ if vol > 0.02 => "high_vol",
            _ if edge > 0.3 => "trending",
            _ if edge < 0.1 => "consolidating",
            _ => "mixed",
        }
    }
}

/// Trait for any probabilistic signal source.
///
/// Implementors: `DualKalmanSignal`, `FrequencySignal`, `OrderBookSignal`,
/// and the existing `MlEnsembleSignal` (gradient-boost wrapper).
pub trait ProbabilisticSignal: Send + Sync {
    /// Return the latest signal output, or `None` if insufficient data.
    fn signal(&self) -> Option<SignalOutput>;
    /// Human-readable source name for logging.
    fn source_name(&self) -> &'static str;
}

pub mod fusion;
