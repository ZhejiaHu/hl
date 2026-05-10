//! Online frequency-domain feature extraction via the Goertzel algorithm.
//!
//! ## Design
//! The Goertzel algorithm computes individual DFT coefficients in O(N) per
//! target frequency, with no future data dependency — safe for rolling/online
//! inference.
//!
//! ## Features produced
//! - **Band powers**: normalised spectral power at target periods.
//! - **Dominant period**: period with highest power.
//! - **Spectral entropy**: 0 = single frequency (trending), 1 = white noise.
//! - **Noise/trend ratio**: short-period power divided by long-period power.
//! - **Dominant phase**: phase of the strongest component.
//!
//! ## How frequency informs the distribution
//! - High spectral entropy → distribution with fatter tails / higher variance.
//! - Low dominant period → noisy / mean-reverting regime (large κ signal).
//! - Low noise/trend ratio → sustained trend (positive drift signal).

/// Target periods (in bars) for Goertzel coefficients.
/// Powers of 2 for efficient FFT compatibility; covers 20 min–5.3 h at 5m bars.
const TARGET_PERIODS: &[usize] = &[4, 8, 16, 32, 64];

const N_BANDS: usize = TARGET_PERIODS.len();

/// Frequency-domain features for one asset at one timestep.
#[derive(Debug, Clone)]
pub struct FrequencyFeatures {
    /// Normalised spectral power at each target period (sums to ≤ 1).
    pub band_powers: [f64; N_BANDS],
    /// Dominant period in bars (period with peak power).
    pub dominant_period: f64,
    /// Spectral entropy ∈ [0, 1].
    pub spectral_entropy: f64,
    /// Ratio of short-period (≤8 bars) to long-period (≥16 bars) power.
    pub noise_trend_ratio: f64,
    /// Phase of the dominant component (radians ∈ [−π, π]).
    pub dominant_phase: f64,
}

impl Default for FrequencyFeatures {
    fn default() -> Self {
        Self {
            band_powers: [0.0; N_BANDS],
            dominant_period: 16.0,
            spectral_entropy: 1.0, // Maximum uncertainty by default.
            noise_trend_ratio: 1.0,
            dominant_phase: 0.0,
        }
    }
}

impl FrequencyFeatures {
    /// Flat feature vector for ML/ensemble input.
    pub fn to_vec(&self) -> Vec<f64> {
        let mut v: Vec<f64> = self.band_powers.to_vec();
        v.push(self.dominant_period / TARGET_PERIODS[N_BANDS - 1] as f64);
        v.push(self.spectral_entropy);
        v.push(self.noise_trend_ratio.min(10.0) / 10.0); // normalised
        v.push(self.dominant_phase / std::f64::consts::PI); // normalised to [-1, 1]
        v
    }

    pub const fn n_features() -> usize {
        N_BANDS + 4
    }

    /// Regime summary derived from frequency features (no discrete label).
    ///
    /// Returns (trend_score, noise_score) ∈ [0,1] each:
    /// - `trend_score` high → low-frequency dominant, sustained price move.
    /// - `noise_score` high → high-frequency dominant, mean-reversion expected.
    pub fn regime_scores(&self) -> (f64, f64) {
        let trend_score = (1.0 - self.spectral_entropy)
            * (1.0 / self.noise_trend_ratio.max(0.1)).min(1.0);
        let noise_score = self.spectral_entropy * self.noise_trend_ratio.min(1.0);
        (trend_score.clamp(0.0, 1.0), noise_score.clamp(0.0, 1.0))
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Online rolling frequency feature extractor.
///
/// Maintains a fixed-length circular buffer. Each new price pushes one sample
/// in; when the buffer is full the Goertzel DFT is recomputed at all target
/// frequencies. Uses de-meaned prices to suppress the DC component.
pub struct FrequencyExtractor {
    window: Vec<f64>,
    window_size: usize,
    head: usize, // next write position
    full: bool,
}

impl FrequencyExtractor {
    pub fn new(window_size: usize) -> Self {
        assert!(window_size >= *TARGET_PERIODS.last().unwrap(), "window too small");
        Self {
            window: vec![0.0; window_size],
            window_size,
            head: 0,
            full: false,
        }
    }

    /// Push one new price and return updated features.
    ///
    /// Returns `None` until the window is full.
    pub fn update(&mut self, price: f64) -> Option<FrequencyFeatures> {
        self.window[self.head] = price;
        self.head = (self.head + 1) % self.window_size;
        if !self.full && self.head == 0 {
            self.full = true;
        }
        if !self.full {
            return None;
        }

        // Build a linear (non-circular) slice in chronological order.
        let signal: Vec<f64> = {
            let mut s = Vec::with_capacity(self.window_size);
            for i in 0..self.window_size {
                let idx = (self.head + i) % self.window_size;
                s.push(self.window[idx]);
            }
            s
        };

        Some(compute_features(&signal))
    }

    /// Latest features without updating (recomputes on current window).
    pub fn latest(&self) -> Option<FrequencyFeatures> {
        if !self.full {
            return None;
        }
        let signal: Vec<f64> = (0..self.window_size)
            .map(|i| self.window[(self.head + i) % self.window_size])
            .collect();
        Some(compute_features(&signal))
    }
}

// ─────────────────────────────────────────────────────────────────────────────

fn compute_features(signal: &[f64]) -> FrequencyFeatures {
    let n = signal.len();
    // De-mean to remove DC offset.
    let mean = signal.iter().sum::<f64>() / n as f64;
    let demeaned: Vec<f64> = signal.iter().map(|&x| x - mean).collect();

    let mut powers = [0.0f64; N_BANDS];
    let mut phases = [0.0f64; N_BANDS];

    for (bi, &period) in TARGET_PERIODS.iter().enumerate() {
        let (re, im) = goertzel(&demeaned, n, period);
        powers[bi] = re * re + im * im;
        phases[bi] = im.atan2(re);
    }

    let total: f64 = powers.iter().sum::<f64>().max(1e-20);
    let mut norm = [0.0f64; N_BANDS];
    for i in 0..N_BANDS {
        norm[i] = powers[i] / total;
    }

    // Dominant period.
    let best_idx = norm
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let dominant_period = TARGET_PERIODS[best_idx] as f64;
    let dominant_phase = phases[best_idx];

    // Spectral entropy H = -Σ p·ln(p) / ln(K).
    let max_h = (N_BANDS as f64).ln();
    let entropy: f64 = norm
        .iter()
        .filter(|&&p| p > 1e-15)
        .map(|&p| -p * p.ln())
        .sum::<f64>();
    let spectral_entropy = (entropy / max_h.max(1e-15)).clamp(0.0, 1.0);

    // Noise/trend ratio.
    let short_power: f64 = norm
        .iter()
        .zip(TARGET_PERIODS)
        .filter(|(_, &p)| p <= 8)
        .map(|(&pw, _)| pw)
        .sum();
    let long_power: f64 = norm
        .iter()
        .zip(TARGET_PERIODS)
        .filter(|(_, &p)| p >= 16)
        .map(|(&pw, _)| pw)
        .sum();
    let noise_trend_ratio = short_power / long_power.max(1e-15);

    FrequencyFeatures {
        band_powers: norm,
        dominant_period,
        spectral_entropy,
        noise_trend_ratio,
        dominant_phase,
    }
}

/// Goertzel algorithm: single DFT bin at frequency k = N / period.
///
/// Returns (real, imag) components of X[k] for the input signal of length N.
/// Complexity: O(N) per call.
fn goertzel(signal: &[f64], n: usize, period: usize) -> (f64, f64) {
    if period == 0 || period > n {
        return (0.0, 0.0);
    }
    let k = n as f64 / period as f64;
    let omega = 2.0 * std::f64::consts::PI * k / n as f64;
    let coeff = 2.0 * omega.cos();

    let mut s_prev2 = 0.0f64;
    let mut s_prev1 = 0.0f64;

    for &x in signal {
        let s = x + coeff * s_prev1 - s_prev2;
        s_prev2 = s_prev1;
        s_prev1 = s;
    }

    let re = s_prev1 - s_prev2 * omega.cos();
    let im = s_prev2 * omega.sin();
    (re, im)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_dominant_period() {
        let mut ext = FrequencyExtractor::new(64);
        // Pure sine at period=16 bars.
        for i in 0..64u64 {
            let price = 100.0 + 2.0 * (2.0 * std::f64::consts::PI * i as f64 / 16.0).sin();
            ext.update(price);
        }
        let feat = ext.latest().unwrap();
        assert!((feat.dominant_period - 16.0).abs() <= 8.0,
            "dominant_period={}", feat.dominant_period);
    }

    #[test]
    fn white_noise_has_high_entropy() {
        let mut ext = FrequencyExtractor::new(64);
        // Fill with values spread across frequencies.
        for i in 0..64u64 {
            let price = 100.0
                + (i as f64 * 0.123).sin()
                + (i as f64 * 0.456).sin()
                + (i as f64 * 0.789).sin()
                + (i as f64 * 1.234).sin();
            ext.update(price);
        }
        let feat = ext.latest().unwrap();
        assert!(feat.spectral_entropy > 0.5,
            "spectral_entropy={}", feat.spectral_entropy);
    }

    #[test]
    fn trend_has_low_noise_ratio() {
        let mut ext = FrequencyExtractor::new(64);
        // Linear trend → energy concentrates at low frequencies.
        for i in 0..64u64 {
            ext.update(100.0 + 0.1 * i as f64);
        }
        let feat = ext.latest().unwrap();
        assert!(feat.noise_trend_ratio < 1.0,
            "noise_trend_ratio={}", feat.noise_trend_ratio);
    }
}
