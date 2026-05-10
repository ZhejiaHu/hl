//! Engle–Granger cointegration test and Ornstein–Uhlenbeck spread modelling.
//!
//! For each pair of assets that pass a cointegration test, we estimate the
//! hedge ratio β via OLS and model the spread S_t = P1_t - β·P2_t as an OU
//! process.  The z-score of the spread is used as a mean-reversion signal.

use anyhow::{bail, Result};

/// Result of the Engle–Granger two-step cointegration test.
#[derive(Debug, Clone)]
pub struct CointegrationResult {
    pub asset1: String,
    pub asset2: String,
    /// OLS hedge ratio (P1 = α + β·P2 + ε).
    pub beta: f64,
    pub alpha: f64,
    /// ADF test statistic on the spread residuals.
    pub adf_stat: f64,
    /// Is the spread stationary at 5% (ADF < -2.86)?
    pub is_cointegrated: bool,
}

/// Ornstein–Uhlenbeck process fitted to a spread series.
#[derive(Debug, Clone)]
pub struct OuProcess {
    pub asset1: String,
    pub asset2: String,
    pub beta: f64,
    pub alpha_const: f64,
    /// Mean reversion speed κ.
    pub kappa: f64,
    /// Long-run mean θ.
    pub theta: f64,
    /// Diffusion σ.
    pub sigma: f64,
    /// Half-life of mean reversion in bars.
    pub half_life: f64,
}

impl OuProcess {
    /// Z-score of the current spread.
    pub fn z_score(&self, price1: f64, price2: f64) -> f64 {
        let spread = price1 - self.alpha_const - self.beta * price2;
        (spread - self.theta) / (self.sigma / (2.0 * self.kappa).sqrt()).max(1e-10)
    }
}

/// Test all pairs in `assets` (by name) for cointegration using price series.
pub fn find_cointegrated_pairs(
    prices: &[(String, Vec<f64>)],
) -> Vec<CointegrationResult> {
    let mut results = Vec::new();
    let n = prices.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let (ref name1, ref p1) = prices[i];
            let (ref name2, ref p2) = prices[j];
            if p1.len() != p2.len() || p1.len() < 30 {
                continue;
            }
            if let Ok(r) = engle_granger(name1, name2, p1, p2) {
                results.push(r);
            }
        }
    }
    results
}

/// Engle–Granger test: OLS regression followed by ADF on residuals.
pub fn engle_granger(
    asset1: &str,
    asset2: &str,
    p1: &[f64],
    p2: &[f64],
) -> Result<CointegrationResult> {
    let n = p1.len();
    if n < 30 {
        bail!("Need ≥ 30 observations");
    }

    let (alpha, beta) = ols(p1, p2);
    let residuals: Vec<f64> = p1.iter().zip(p2).map(|(&y, &x)| y - alpha - beta * x).collect();
    let adf_stat = adf_test(&residuals);

    // 5% critical value for ADF without trend ≈ -2.86.
    Ok(CointegrationResult {
        asset1: asset1.to_string(),
        asset2: asset2.to_string(),
        beta,
        alpha,
        adf_stat,
        is_cointegrated: adf_stat < -2.86,
    })
}

/// Fit an OU process to a spread series via discrete-time OLS.
///
/// The discrete OU equation: ΔS_t = κ·(θ - S_{t-1})·Δt + σ·ε_t
/// Regressing ΔS_t on S_{t-1} gives parameter estimates.
pub fn fit_ou(coint: &CointegrationResult, p1: &[f64], p2: &[f64]) -> Result<OuProcess> {
    let spreads: Vec<f64> = p1
        .iter()
        .zip(p2)
        .map(|(&y, &x)| y - coint.alpha - coint.beta * x)
        .collect();

    let n = spreads.len();
    if n < 10 {
        bail!("Need ≥ 10 spread observations");
    }

    let s_lag = &spreads[..n - 1];
    let ds: Vec<f64> = spreads.windows(2).map(|w| w[1] - w[0]).collect();

    // Regress ΔS on S_{t-1}: ΔS = a + b·S_{t-1}
    let (a, b) = ols(&ds, s_lag);

    // b = -κ·Δt → κ = -b (Δt = 1 bar).
    let kappa = (-b).max(1e-8);
    // a = κ·θ → θ = a/κ.
    let theta = a / kappa;

    let residuals: Vec<f64> = ds.iter().zip(s_lag).map(|(&d, &s)| d - a - b * s).collect();
    let sigma = residuals.iter().map(|r| r * r).sum::<f64>().sqrt() / (n as f64).sqrt();
    let half_life = 2.0_f64.ln() / kappa;

    Ok(OuProcess {
        asset1: coint.asset1.clone(),
        asset2: coint.asset2.clone(),
        beta: coint.beta,
        alpha_const: coint.alpha,
        kappa,
        theta,
        sigma: sigma.max(1e-10),
        half_life,
    })
}

// ─── private helpers ────────────────────────────────────────────────────────

/// Simple OLS: y = α + β·x, returns (α, β).
fn ols(y: &[f64], x: &[f64]) -> (f64, f64) {
    let n = y.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xx: f64 = x.iter().map(|&xi| xi * xi).sum();
    let sum_xy: f64 = x.iter().zip(y).map(|(&xi, &yi)| xi * yi).sum();
    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-12 {
        return (sum_y / n, 0.0);
    }
    let beta = (n * sum_xy - sum_x * sum_y) / denom;
    let alpha = (sum_y - beta * sum_x) / n;
    (alpha, beta)
}

/// Augmented Dickey–Fuller test statistic (lag=1, no trend).
fn adf_test(series: &[f64]) -> f64 {
    let n = series.len();
    if n < 10 {
        return 0.0;
    }
    let ds: Vec<f64> = series.windows(2).map(|w| w[1] - w[0]).collect();
    let s_lag: Vec<f64> = series[..n - 1].to_vec();

    let (_, b) = ols(&ds, &s_lag);

    // Compute standard error of b.
    let predictions: Vec<f64> = s_lag.iter().zip(&ds).map(|(&sl, _)| b * sl).collect();
    let residuals: Vec<f64> = ds.iter().zip(&predictions).map(|(&d, &p)| d - p).collect();
    let rss: f64 = residuals.iter().map(|r| r * r).sum();
    let s_xx: f64 = s_lag.iter().map(|&s| s * s).sum();
    let se = if s_xx < 1e-12 {
        return 0.0;
    } else {
        (rss / (n - 2) as f64 / s_xx).sqrt()
    };

    b / se
}
