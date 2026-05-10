//! Bayesian model combination for the signal ensemble.
//!
//! ## Algorithm
//! Each signal source m has weight w_m ∝ exp(EMA[log p(z_t | m)]).
//! The combined distribution is a finite mixture:
//!   p_up       = Σ_m w_m · p_up_m
//!   μ_mixture  = Σ_m w_m · μ_m
//!   σ_mixture² = Σ_m w_m · (σ_m² + (μ_m − μ_mix)²)   [mixture variance]
//!
//! Weights are updated online via an exponential moving average of log-evidence
//! so that better-performing models receive higher weight over time.
//!
//! ## Tuning parameters
//! - `evidence_decay` ∈ (0,1]: 1 = no forgetting (equal weighting), <1 = exponential forgetting.
//!   Default 0.99 ≈ 100-bar half-life, appropriate for 5-minute intraday data.
//! - `min_weight`: floor on any single model's weight to preserve diversity.

use super::{EnsembleDistribution, SignalOutput};

/// Bayesian fusion of multiple probabilistic signal sources.
///
/// Maintains per-source log-evidence accumulators and derives weights.
pub struct BayesianFusion {
    /// Source names in registration order.
    sources: Vec<&'static str>,
    /// Exponential moving average of log-evidence per source.
    log_evidence_ema: Vec<f64>,
    /// EMA decay factor per update.
    evidence_decay: f64,
    /// Minimum weight floor (prevents any source from being ignored).
    min_weight: f64,
}

impl BayesianFusion {
    /// Create with configurable parameters.
    ///
    /// - `n_sources`: number of signal sources to combine.
    /// - `evidence_decay`: EMA forgetting factor (0.99 = slow forgetting).
    /// - `min_weight`: minimum weight fraction per source (e.g. 0.05).
    pub fn new(n_sources: usize, evidence_decay: f64, min_weight: f64) -> Self {
        Self {
            sources: Vec::with_capacity(n_sources),
            log_evidence_ema: vec![0.0; n_sources],
            evidence_decay,
            min_weight: min_weight.clamp(0.0, 1.0 / n_sources as f64),
        }
    }

    /// Default: 3 sources, slow forgetting, 5% minimum weight.
    pub fn default_three_source() -> Self {
        Self::new(3, 0.99, 0.05)
    }

    /// Register a source name (call once per source in order).
    pub fn register_source(&mut self, name: &'static str) {
        if self.sources.len() < self.log_evidence_ema.len() {
            self.sources.push(name);
        }
    }

    /// Fuse a slice of signal outputs (one per registered source, same order).
    ///
    /// Missing signals (None) are skipped; their weights are redistributed.
    /// Returns `None` if no signals are available.
    pub fn fuse(&mut self, signals: &[Option<SignalOutput>]) -> Option<EnsembleDistribution> {
        let n = self.log_evidence_ema.len().min(signals.len());
        if n == 0 {
            return None;
        }

        // Update log-evidence EMA for each source.
        for i in 0..n {
            if let Some(sig) = &signals[i] {
                let le = sig.log_evidence.clamp(-50.0, 0.0);
                self.log_evidence_ema[i] =
                    self.evidence_decay * self.log_evidence_ema[i] + (1.0 - self.evidence_decay) * le;
            }
        }

        // Compute weights: softmax over log-evidence EMA (only for active signals).
        let active: Vec<bool> = (0..n).map(|i| signals[i].is_some()).collect();
        let active_count = active.iter().filter(|&&a| a).count();
        if active_count == 0 {
            return None;
        }

        let raw_weights = self.compute_weights(&active);
        let weights: Vec<f64> = (0..n)
            .map(|i| if active[i] { raw_weights[i] } else { 0.0 })
            .collect();

        // Mixture moments.
        let mut p_up = 0.0f64;
        let mut p_down = 0.0f64;
        let mut mu_mix = 0.0f64;
        let mut conf_mix = 0.0f64;

        for i in 0..n {
            if let Some(sig) = &signals[i] {
                let w = weights[i];
                p_up += w * sig.p_up;
                p_down += w * sig.p_down;
                mu_mix += w * sig.predictive_mean;
                conf_mix += w * sig.confidence();
            }
        }

        // Mixture variance = Σ w_i · (σ_i² + (μ_i − μ_mix)²).
        let sigma2_mix: f64 = (0..n)
            .filter_map(|i| signals[i].as_ref().map(|s| (i, s)))
            .map(|(i, s)| {
                let var = s.predictive_std * s.predictive_std;
                let bias2 = (s.predictive_mean - mu_mix).powi(2);
                weights[i] * (var + bias2)
            })
            .sum();

        let named_weights: Vec<(&'static str, f64)> = (0..n)
            .map(|i| {
                let name = self.sources.get(i).copied().unwrap_or("unknown");
                (name, weights[i])
            })
            .collect();

        Some(EnsembleDistribution {
            p_up: p_up.clamp(0.0, 1.0),
            p_down: p_down.clamp(0.0, 1.0),
            predictive_mean: mu_mix,
            predictive_std: sigma2_mix.sqrt().max(1e-10),
            confidence: conf_mix.clamp(0.0, 1.0),
            weights: named_weights,
        })
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn compute_weights(&self, active: &[bool]) -> Vec<f64> {
        let n = active.len();
        // Subtract max for numerical stability in softmax.
        let max_le = active
            .iter()
            .zip(&self.log_evidence_ema)
            .filter(|(&a, _)| a)
            .map(|(_, &le)| le)
            .fold(f64::NEG_INFINITY, f64::max);

        let exp_vals: Vec<f64> = (0..n)
            .map(|i| {
                if active[i] {
                    (self.log_evidence_ema[i] - max_le).exp()
                } else {
                    0.0
                }
            })
            .collect();

        let sum_exp: f64 = exp_vals.iter().sum::<f64>().max(1e-30);
        let raw: Vec<f64> = exp_vals.iter().map(|&e| e / sum_exp).collect();

        // Apply minimum weight floor and renormalise.
        let floor = self.min_weight;
        let n_active = active.iter().filter(|&&a| a).count() as f64;
        let floor_total = floor * n_active;

        if floor_total >= 1.0 {
            // Uniform fallback if floor would exceed total weight budget.
            return (0..n)
                .map(|i| if active[i] { 1.0 / n_active } else { 0.0 })
                .collect();
        }

        let remaining = 1.0 - floor_total;
        (0..n)
            .map(|i| {
                if active[i] {
                    floor + remaining * raw[i]
                } else {
                    0.0
                }
            })
            .collect()
    }
}

// ─── Signal adapters ──────────────────────────────────────────────────────────

use crate::estimation::StatePosterior;

/// Wrap a `StatePosterior` as a `SignalOutput` from the Kalman estimator.
pub fn kalman_signal(posterior: &StatePosterior) -> SignalOutput {
    use crate::estimation::standard_normal_cdf;
    let mu = posterior.predictive_mean();
    let sigma = posterior.predictive_std();
    let p_up = standard_normal_cdf(mu / sigma);
    let p_down = standard_normal_cdf(-mu / sigma);

    SignalOutput {
        source: "dual_kalman",
        p_up: p_up.clamp(0.0, 1.0),
        p_down: p_down.clamp(0.0, 1.0),
        predictive_mean: mu,
        predictive_std: sigma,
        log_evidence: posterior.log_predictive,
    }
}

use crate::features::frequency::FrequencyFeatures;

/// Derive a signal from frequency features.
///
/// High long-period power with low entropy → directional bias toward trend.
/// High spectral entropy → reduced confidence.
pub fn frequency_signal(freq: &FrequencyFeatures, current_velocity: f64) -> SignalOutput {
    use crate::estimation::standard_normal_cdf;
    let (trend_score, noise_score) = freq.regime_scores();

    // Translate trend score and velocity into a directional bias.
    let directional_bias = trend_score * current_velocity.signum() * trend_score.sqrt();
    let implied_std = (0.005 + noise_score * 0.02).max(1e-10);

    let p_up = standard_normal_cdf(directional_bias / implied_std).clamp(0.01, 0.99);
    let p_down = standard_normal_cdf(-directional_bias / implied_std).clamp(0.01, 0.99);

    // Confidence decays with spectral entropy.
    let conf_weight = 1.0 - freq.spectral_entropy;
    // Log-evidence: lower entropy → better predictive power.
    let log_evidence = -freq.spectral_entropy.max(1e-5).ln() * conf_weight - 1.0;

    SignalOutput {
        source: "frequency",
        p_up,
        p_down,
        predictive_mean: directional_bias * implied_std,
        predictive_std: implied_std,
        log_evidence: log_evidence.clamp(-20.0, 0.0),
    }
}

use crate::orderbook::hawkes::HawkesProcess;

/// Derive a signal from Hawkes process order-book dynamics.
///
/// Intensity OFI (order-flow imbalance) from background rates encodes the
/// net buy/sell pressure. Cross-excitation ratio encodes stop-loss cascade risk.
pub fn orderbook_signal(hawkes: &HawkesProcess) -> SignalOutput {
    use crate::estimation::standard_normal_cdf;
    let ofi = hawkes.intensity_ofi().clamp(-1.0, 1.0);
    let cross_ratio = hawkes.cross_excitation_ratio().clamp(0.0, 5.0);

    // High cross-excitation → fatter tails / lower confidence.
    let tail_penalty = (cross_ratio / 5.0).min(0.5);
    let implied_std = (0.003 + tail_penalty * 0.015).max(1e-10);

    let p_up = standard_normal_cdf(ofi / implied_std * 0.5).clamp(0.01, 0.99);
    let p_down = standard_normal_cdf(-ofi / implied_std * 0.5).clamp(0.01, 0.99);

    // Log-evidence: higher OFI magnitude → better prediction confidence.
    let log_evidence = -0.5 * (1.0 - ofi.abs()) - tail_penalty;

    SignalOutput {
        source: "orderbook",
        p_up,
        p_down,
        predictive_mean: ofi * implied_std,
        predictive_std: implied_std,
        log_evidence: log_evidence.clamp(-20.0, 0.0),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_signal(source: &'static str, p_up: f64, p_down: f64, le: f64) -> SignalOutput {
        SignalOutput {
            source,
            p_up,
            p_down,
            predictive_mean: p_up - p_down,
            predictive_std: 0.01,
            log_evidence: le,
        }
    }

    #[test]
    fn fusion_averages_signals() {
        let mut fusion = BayesianFusion::new(2, 0.99, 0.0);
        fusion.register_source("a");
        fusion.register_source("b");

        let s1 = make_signal("a", 0.6, 0.2, -1.0);
        let s2 = make_signal("b", 0.4, 0.4, -2.0);

        let result = fusion.fuse(&[Some(s1), Some(s2)]).unwrap();
        assert!(result.p_up > 0.4 && result.p_up < 0.7);
    }

    #[test]
    fn fusion_handles_missing_source() {
        let mut fusion = BayesianFusion::new(3, 0.99, 0.0);
        fusion.register_source("a");
        fusion.register_source("b");
        fusion.register_source("c");

        let s1 = make_signal("a", 0.7, 0.1, -0.5);
        let result = fusion.fuse(&[Some(s1), None, None]).unwrap();
        assert!((result.p_up - 0.7).abs() < 0.01);
    }

    #[test]
    fn better_evidence_gets_higher_weight() {
        let mut fusion = BayesianFusion::new(2, 0.5, 0.0);
        fusion.register_source("good");
        fusion.register_source("bad");

        // Feed 20 steps: good model always has better evidence.
        for _ in 0..20 {
            let g = make_signal("good", 0.8, 0.1, -0.1);
            let b = make_signal("bad", 0.5, 0.4, -5.0);
            fusion.fuse(&[Some(g), Some(b)]);
        }
        // Final fuse to check weights.
        let g = make_signal("good", 0.8, 0.1, -0.1);
        let b = make_signal("bad", 0.5, 0.4, -5.0);
        let result = fusion.fuse(&[Some(g), Some(b)]).unwrap();
        let w_good = result.weights[0].1;
        let w_bad = result.weights[1].1;
        assert!(w_good > w_bad, "good={} bad={}", w_good, w_bad);
    }
}
