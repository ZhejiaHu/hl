//! Return distribution fitting and model selection.
//!
//! Three candidate distributions are fitted to log-returns and ranked by AIC:
//!
//!   ┌──────────────┬────────┬────────────────────────────────────────────┐
//!   │ Distribution │ Params │ Best suited for                            │
//!   ├──────────────┼────────┼────────────────────────────────────────────┤
//!   │ Student-t    │   3    │ Symmetric heavy tails (most crypto assets) │
//!   │ Cauchy       │   2    │ Extremely heavy / infinite-variance tails  │
//!   │ Discrete     │ k−1*   │ Skewed / bimodal / non-parametric shape    │
//!   └──────────────┴────────┴────────────────────────────────────────────┘
//!   * k = number of non-empty histogram bins.
//!
//! `ReturnDist::fit_best` tries all three and returns the winner.  Every
//! variant exposes a uniform interface: `log_pdf`, `cdf`, `var`, `cvar`.

use anyhow::{bail, Result};

// ── Shared math utilities ────────────────────────────────────────────────────

fn sample_mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn sample_std(xs: &[f64], mu: f64) -> f64 {
    let var = xs.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / xs.len() as f64;
    var.sqrt()
}

/// Median of an unsorted slice (allocates a sorted copy).
fn median(xs: &[f64]) -> f64 {
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 0 {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    } else {
        s[n / 2]
    }
}

fn lgamma(x: f64) -> f64 {
    statrs::function::gamma::ln_gamma(x)
}

// ── Student-t ────────────────────────────────────────────────────────────────

/// Standardised Student-t distribution fitted by EM-MLE.
#[derive(Debug, Clone)]
pub struct StudentT {
    /// Location (mean).
    pub mu: f64,
    /// Scale (std-dev-like spread).
    pub sigma: f64,
    /// Degrees of freedom (>2 for finite variance; typical crypto: 3–6).
    pub nu: f64,
    pub log_likelihood: f64,
    pub ks_stat: f64,
}

impl StudentT {
    /// Fit to `returns` via EM-MLE (≥ 30 observations required).
    pub fn fit(returns: &[f64]) -> Result<Self> {
        let n = returns.len();
        if n < 30 {
            bail!("StudentT needs ≥ 30 observations (have {})", n);
        }

        let mu = sample_mean(returns);
        let mut sigma = sample_std(returns, mu).max(1e-10);
        let mut nu = 5.0f64;

        for _ in 0..200 {
            // E-step: weights w_i = (ν+1) / (ν + z_i²).
            let weights: Vec<f64> = returns
                .iter()
                .map(|&r| {
                    let z = (r - mu) / sigma;
                    (nu + 1.0) / (nu + z * z)
                })
                .collect();

            // M-step for σ².
            let sigma2 = returns
                .iter()
                .zip(&weights)
                .map(|(&r, &w)| w * (r - mu).powi(2))
                .sum::<f64>()
                / n as f64;
            sigma = sigma2.sqrt().max(1e-10);

            // M-step for ν via digamma fixed-point.
            let rhs: f64 = 1.0
                + weights
                    .iter()
                    .zip(returns.iter())
                    .map(|(&w, &r)| {
                        let z = (r - mu) / sigma;
                        w - (z * z * nu / (nu + z * z)).ln()
                            - w * (z * z * nu / (nu + z * z))
                    })
                    .sum::<f64>()
                    / n as f64;
            nu = nu_from_fixed_point(rhs).max(2.1);
        }

        let ll = t_log_likelihood(returns, mu, sigma, nu);
        let ks = t_ks_statistic(returns, mu, sigma, nu);
        Ok(Self { mu, sigma, nu, log_likelihood: ll, ks_stat: ks })
    }

    /// AIC = 2k − 2·LL, k = 3 (μ, σ, ν).
    pub fn aic(&self) -> f64 {
        2.0 * 3.0 - 2.0 * self.log_likelihood
    }

    pub fn log_pdf(&self, x: f64) -> f64 {
        t_log_pdf(x, self.mu, self.sigma, self.nu)
    }

    pub fn cdf(&self, x: f64) -> f64 {
        t_cdf_std((x - self.mu) / self.sigma, self.nu)
    }

    /// VaR at confidence `alpha` (e.g. 0.95 → 5th-percentile return).
    pub fn var(&self, alpha: f64) -> f64 {
        let t_q = t_quantile_bisect(self.nu, 1.0 - alpha);
        self.mu + self.sigma * t_q
    }

    /// Expected shortfall (CVaR) at `alpha`.
    pub fn cvar(&self, alpha: f64) -> f64 {
        let v = self.var(alpha);
        let z = (v - self.mu) / self.sigma;
        let pdf_z = t_pdf_std(z, self.nu);
        let cdf_z = t_cdf_std(z, self.nu);
        if cdf_z < 1e-15 {
            return v;
        }
        self.mu
            + self.sigma
                * (-pdf_z * (self.nu + z * z) / ((self.nu - 1.0) * cdf_z))
    }
}

// ── Cauchy ────────────────────────────────────────────────────────────────────

/// Cauchy distribution fitted by alternating MLE (IRLS for x₀, Newton for γ).
///
/// Cauchy has undefined mean and infinite variance — it captures asset returns
/// where even Student-t tail estimates are too thin.
#[derive(Debug, Clone)]
pub struct CauchyDist {
    /// Location parameter (median of the distribution).
    pub x0: f64,
    /// Scale parameter (half-width at half-maximum).
    pub gamma: f64,
    pub log_likelihood: f64,
    pub ks_stat: f64,
}

impl CauchyDist {
    /// Fit to `returns` via alternating MLE (≥ 30 observations required).
    ///
    /// Location update: IRLS weighted mean (Cauchy weights 1/(γ²+(x−x₀)²)).
    /// Scale update: Newton step on the MLE condition Σ γ²/(γ²+sᵢ) = n/2.
    pub fn fit(returns: &[f64]) -> Result<Self> {
        let n = returns.len();
        if n < 30 {
            bail!("CauchyDist needs ≥ 30 observations (have {})", n);
        }

        // Initialise: median for x0, IQR/2 for γ.
        let mut sorted = returns.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut x0 = median(returns);
        let q1 = sorted[n / 4];
        let q3 = sorted[(3 * n) / 4];
        let mut gamma = ((q3 - q1) / 2.0).max(1e-10);

        let mut best_ll = f64::NEG_INFINITY;
        let mut best = (x0, gamma);

        for _ in 0..300 {
            let s: Vec<f64> = returns.iter().map(|&r| (r - x0).powi(2)).collect();

            // IRLS update for x₀: Cauchy-weighted mean.
            let w: Vec<f64> = s.iter().map(|&si| 1.0 / (gamma * gamma + si)).collect();
            let w_sum: f64 = w.iter().sum();
            if w_sum > 1e-15 {
                x0 = returns.iter().zip(&w).map(|(&r, &wi)| r * wi).sum::<f64>() / w_sum;
            }

            // Newton step for u = γ² via MLE condition Σ u/(u+sᵢ) = n/2.
            let u = gamma * gamma;
            let h: f64 =
                s.iter().map(|&si| u / (u + si)).sum::<f64>() - n as f64 / 2.0;
            let hp: f64 = s.iter().map(|&si| si / (u + si).powi(2)).sum::<f64>();
            if hp.abs() > 1e-15 {
                let u_new = (u - h / hp).max(1e-20);
                gamma = u_new.sqrt();
            }

            let ll = cauchy_log_likelihood(returns, x0, gamma);
            if ll > best_ll {
                best_ll = ll;
                best = (x0, gamma);
            }
        }

        (x0, gamma) = best;
        let ks = cauchy_ks_statistic(returns, x0, gamma);
        Ok(Self { x0, gamma, log_likelihood: best_ll, ks_stat: ks })
    }

    /// AIC = 2k − 2·LL, k = 2 (x₀, γ).
    pub fn aic(&self) -> f64 {
        2.0 * 2.0 - 2.0 * self.log_likelihood
    }

    pub fn log_pdf(&self, x: f64) -> f64 {
        cauchy_log_pdf(x, self.x0, self.gamma)
    }

    /// CDF: F(x) = 0.5 + arctan((x−x₀)/γ) / π.
    pub fn cdf(&self, x: f64) -> f64 {
        0.5 + ((x - self.x0) / self.gamma).atan() / std::f64::consts::PI
    }

    /// VaR at confidence `alpha` (exact Cauchy quantile).
    pub fn var(&self, alpha: f64) -> f64 {
        let p = 1.0 - alpha;
        self.x0 + self.gamma * (std::f64::consts::PI * (p - 0.5)).tan()
    }

    /// Bounded CVaR at `alpha`.
    ///
    /// Cauchy has no finite mean, so the integral ∫_{-∞}^{VaR} x·f(x)dx
    /// diverges.  We truncate the lower bound at the 0.001 quantile —
    /// equivalent to assuming returns beyond that threshold are floored at the
    /// exchange circuit-breaker level, matching practical risk management.
    ///
    /// Closed-form antiderivative (from -∞ perspective):
    ///   ∫_a^b x·f(x)dx = x₀·[F(b)−F(a)] + γ/(2π)·[ln(1+(b−x₀)²/γ²) − ln(1+(a−x₀)²/γ²)]
    pub fn cvar(&self, alpha: f64) -> f64 {
        let q = self.var(alpha);
        let p_var = 1.0 - alpha; // F(q)

        // Lower truncation at 0.001 quantile.
        let p_low = 0.001f64;
        let a = self.x0 + self.gamma * (std::f64::consts::PI * (p_low - 0.5)).tan();

        let z_q = (q - self.x0) / self.gamma;
        let z_a = (a - self.x0) / self.gamma;

        // ∫_a^q x·f(x)dx
        let integral = self.x0 * (p_var - p_low)
            + self.gamma / (2.0 * std::f64::consts::PI)
                * ((1.0 + z_q * z_q).ln() - (1.0 + z_a * z_a).ln());

        if p_var < 1e-15 {
            return q;
        }
        integral / p_var
    }
}

// ── Discrete (histogram) distribution ────────────────────────────────────────

/// Non-parametric discrete distribution: Laplace-smoothed histogram density.
///
/// Bin width follows Scott's rule (3.5·σ·n^{-1/3}), capped at 5–100 bins.
/// Laplace smoothing (α = 0.5) prevents log(0) for empty bins while keeping
/// the effective parameter count equal to the number of *filled* bins − 1.
#[derive(Debug, Clone)]
pub struct DiscreteDist {
    /// Bin edges (length = n_bins + 1); uniform spacing.
    pub edges: Vec<f64>,
    pub bin_width: f64,
    /// Smoothed probability mass per bin (sums to 1.0).
    pub probs: Vec<f64>,
    /// Probability density per bin (probs / bin_width).
    pub densities: Vec<f64>,
    pub log_likelihood: f64,
    pub ks_stat: f64,
    /// Filled bins (original count > 0) minus 1 — used for AIC.
    n_free_params: usize,
}

impl DiscreteDist {
    /// Fit histogram to `returns` using Scott's rule binning (≥ 30 required).
    pub fn fit(returns: &[f64]) -> Result<Self> {
        let n = returns.len();
        if n < 30 {
            bail!("DiscreteDist needs ≥ 30 observations (have {})", n);
        }

        let mu = sample_mean(returns);
        let sigma = sample_std(returns, mu).max(1e-10);

        // Scott's rule bin width.
        let raw_width = 3.5 * sigma * (n as f64).powf(-1.0 / 3.0);

        let min_val = returns.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = returns.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Extend range by one raw width on each side so boundary observations
        // always land inside the grid.
        let lo = min_val - raw_width;
        let hi = max_val + raw_width;

        let n_bins_raw = ((hi - lo) / raw_width).ceil() as usize;
        let n_bins = n_bins_raw.clamp(5, 100);

        // Recompute width so bins tile [lo, hi] exactly.
        let bin_width = (hi - lo) / n_bins as f64;

        let edges: Vec<f64> = (0..=n_bins).map(|i| lo + i as f64 * bin_width).collect();

        // Count observations per bin.
        let mut counts = vec![0usize; n_bins];
        for &r in returns {
            let idx = ((r - lo) / bin_width) as usize;
            counts[idx.min(n_bins - 1)] += 1;
        }
        let n_filled = counts.iter().filter(|&&c| c > 0).count();

        // Laplace smoothing (α = 0.5 per bin).
        let alpha_lap = 0.5;
        let total = n as f64 + alpha_lap * n_bins as f64;
        let probs: Vec<f64> =
            counts.iter().map(|&c| (c as f64 + alpha_lap) / total).collect();
        let densities: Vec<f64> = probs.iter().map(|&p| p / bin_width).collect();

        // Log-likelihood as a density estimator.
        let ll = returns
            .iter()
            .map(|&r| {
                let idx = ((r - lo) / bin_width) as usize;
                densities[idx.min(n_bins - 1)].ln()
            })
            .sum::<f64>();

        // KS statistic against the piecewise-linear CDF.
        let ks = {
            let mut sorted = returns.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let empirical = (i + 1) as f64 / n as f64;
                    let theoretical = discrete_cdf(&edges, &probs, bin_width, x);
                    (empirical - theoretical).abs()
                })
                .fold(0.0f64, f64::max)
        };

        Ok(Self {
            edges,
            bin_width,
            probs,
            densities,
            log_likelihood: ll,
            ks_stat: ks,
            n_free_params: n_filled.saturating_sub(1),
        })
    }

    /// AIC = 2k − 2·LL, k = number of filled bins − 1.
    pub fn aic(&self) -> f64 {
        2.0 * self.n_free_params as f64 - 2.0 * self.log_likelihood
    }

    /// Log probability density at `x` (returns −∞ if outside grid).
    pub fn log_pdf(&self, x: f64) -> f64 {
        let lo = self.edges[0];
        let hi = *self.edges.last().unwrap();
        if x < lo || x > hi {
            return f64::NEG_INFINITY;
        }
        let idx = ((x - lo) / self.bin_width) as usize;
        self.densities[idx.min(self.densities.len() - 1)].ln()
    }

    /// Piecewise-linear CDF.
    pub fn cdf(&self, x: f64) -> f64 {
        discrete_cdf(&self.edges, &self.probs, self.bin_width, x)
    }

    /// VaR at confidence `alpha` (linear interpolation within the bin).
    pub fn var(&self, alpha: f64) -> f64 {
        let target = 1.0 - alpha;
        let mut cumprob = 0.0f64;
        for (i, &p) in self.probs.iter().enumerate() {
            let next = cumprob + p;
            if next >= target {
                let frac = if p > 1e-15 { (target - cumprob) / p } else { 0.5 };
                return self.edges[i] + frac * self.bin_width;
            }
            cumprob = next;
        }
        *self.edges.last().unwrap()
    }

    /// Expected shortfall: probability-weighted average of bin midpoints below VaR.
    pub fn cvar(&self, alpha: f64) -> f64 {
        let q = self.var(alpha);
        let mut weighted = 0.0f64;
        let mut prob_sum = 0.0f64;

        for i in 0..self.probs.len() {
            let left = self.edges[i];
            let right = self.edges[i + 1];
            let p = self.probs[i];

            if right <= q {
                // Full bin below VaR.
                weighted += p * (left + right) / 2.0;
                prob_sum += p;
            } else if left < q {
                // Bin straddles VaR — take the portion below.
                let frac = (q - left) / self.bin_width;
                weighted += p * frac * (left + q) / 2.0;
                prob_sum += p * frac;
            }
        }

        if prob_sum < 1e-15 {
            return q;
        }
        weighted / prob_sum
    }
}

// ── Model selection ───────────────────────────────────────────────────────────

/// Best-fit return distribution selected by AIC across Student-t, Cauchy, and
/// the non-parametric discrete histogram.
///
/// ```
/// let dist = ReturnDist::fit_best(&returns)?;
/// println!("{} won (AIC={:.1})", dist.name(), dist.aic());
/// let var_95 = dist.var(0.95);
/// let cvar_95 = dist.cvar(0.95);
/// ```
#[derive(Debug, Clone)]
pub enum ReturnDist {
    StudentT(StudentT),
    Cauchy(CauchyDist),
    Discrete(DiscreteDist),
}

impl ReturnDist {
    /// Fit all three distributions and return the one with the lowest AIC.
    ///
    /// At least one fit must succeed; returns an error only if all three fail.
    pub fn fit_best(returns: &[f64]) -> Result<Self> {
        let mut candidates: Vec<(f64, ReturnDist)> = Vec::new();

        if let Ok(t) = StudentT::fit(returns) {
            candidates.push((t.aic(), ReturnDist::StudentT(t)));
        }
        if let Ok(c) = CauchyDist::fit(returns) {
            candidates.push((c.aic(), ReturnDist::Cauchy(c)));
        }
        if let Ok(d) = DiscreteDist::fit(returns) {
            candidates.push((d.aic(), ReturnDist::Discrete(d)));
        }

        if candidates.is_empty() {
            bail!("All distribution fits failed — need ≥ 30 observations");
        }

        // Lowest AIC wins.
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        Ok(candidates.remove(0).1)
    }

    pub fn name(&self) -> &'static str {
        match self {
            ReturnDist::StudentT(_) => "Student-t",
            ReturnDist::Cauchy(_) => "Cauchy",
            ReturnDist::Discrete(_) => "Discrete",
        }
    }

    pub fn log_pdf(&self, x: f64) -> f64 {
        match self {
            ReturnDist::StudentT(d) => d.log_pdf(x),
            ReturnDist::Cauchy(d) => d.log_pdf(x),
            ReturnDist::Discrete(d) => d.log_pdf(x),
        }
    }

    pub fn cdf(&self, x: f64) -> f64 {
        match self {
            ReturnDist::StudentT(d) => d.cdf(x),
            ReturnDist::Cauchy(d) => d.cdf(x),
            ReturnDist::Discrete(d) => d.cdf(x),
        }
    }

    /// VaR at confidence `alpha` (e.g. 0.95 → worst 5% left-tail return).
    pub fn var(&self, alpha: f64) -> f64 {
        match self {
            ReturnDist::StudentT(d) => d.var(alpha),
            ReturnDist::Cauchy(d) => d.var(alpha),
            ReturnDist::Discrete(d) => d.var(alpha),
        }
    }

    /// Expected shortfall (CVaR) at `alpha`.
    pub fn cvar(&self, alpha: f64) -> f64 {
        match self {
            ReturnDist::StudentT(d) => d.cvar(alpha),
            ReturnDist::Cauchy(d) => d.cvar(alpha),
            ReturnDist::Discrete(d) => d.cvar(alpha),
        }
    }

    pub fn log_likelihood(&self) -> f64 {
        match self {
            ReturnDist::StudentT(d) => d.log_likelihood,
            ReturnDist::Cauchy(d) => d.log_likelihood,
            ReturnDist::Discrete(d) => d.log_likelihood,
        }
    }

    pub fn ks_stat(&self) -> f64 {
        match self {
            ReturnDist::StudentT(d) => d.ks_stat,
            ReturnDist::Cauchy(d) => d.ks_stat,
            ReturnDist::Discrete(d) => d.ks_stat,
        }
    }

    pub fn aic(&self) -> f64 {
        match self {
            ReturnDist::StudentT(d) => d.aic(),
            ReturnDist::Cauchy(d) => d.aic(),
            ReturnDist::Discrete(d) => d.aic(),
        }
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

// ── Student-t internals ──────────────────────────────────────────────────────

fn digamma(x: f64) -> f64 {
    if x < 6.0 {
        return digamma(x + 1.0) - 1.0 / x;
    }
    let r = 1.0 / x;
    x.ln() - 0.5 * r - r.powi(2) * (1.0 / 12.0 - r.powi(2) * (1.0 / 120.0 - r.powi(2) / 252.0))
}

fn trigamma(x: f64) -> f64 {
    if x < 6.0 {
        return trigamma(x + 1.0) + 1.0 / x.powi(2);
    }
    let r = 1.0 / x;
    r + 0.5 * r.powi(2) + r.powi(3) / 6.0 - r.powi(5) / 30.0
}

fn nu_from_fixed_point(target: f64) -> f64 {
    let mut nu = 5.0f64;
    for _ in 0..50 {
        let f = digamma((nu + 1.0) / 2.0) - digamma(nu / 2.0) - target;
        let df = 0.5 * trigamma((nu + 1.0) / 2.0) - 0.5 * trigamma(nu / 2.0);
        if df.abs() < 1e-15 {
            break;
        }
        let step = f / df;
        nu = (nu - step).max(2.1);
        if step.abs() < 1e-8 {
            break;
        }
    }
    nu
}

fn t_log_pdf(x: f64, mu: f64, sigma: f64, nu: f64) -> f64 {
    let z = (x - mu) / sigma;
    lgamma((nu + 1.0) / 2.0)
        - lgamma(nu / 2.0)
        - 0.5 * (std::f64::consts::PI * nu).ln()
        - sigma.ln()
        - (nu + 1.0) / 2.0 * (1.0 + z * z / nu).ln()
}

fn t_log_likelihood(returns: &[f64], mu: f64, sigma: f64, nu: f64) -> f64 {
    returns.iter().map(|&r| t_log_pdf(r, mu, sigma, nu)).sum()
}

fn t_pdf_std(t: f64, nu: f64) -> f64 {
    let lc = lgamma((nu + 1.0) / 2.0)
        - lgamma(nu / 2.0)
        - 0.5 * (std::f64::consts::PI * nu).ln();
    lc.exp() * (1.0 + t * t / nu).powf(-(nu + 1.0) / 2.0)
}

fn t_cdf_std(t: f64, nu: f64) -> f64 {
    let x = nu / (nu + t * t);
    let ib = regularised_incomplete_beta(nu / 2.0, 0.5, x);
    if t < 0.0 { 0.5 * ib } else { 1.0 - 0.5 * ib }
}

fn t_quantile_bisect(nu: f64, p: f64) -> f64 {
    let mut lo = -50.0f64;
    let mut hi = 0.0f64;
    for _ in 0..120 {
        let mid = (lo + hi) / 2.0;
        if t_cdf_std(mid, nu) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}

fn t_ks_statistic(returns: &[f64], mu: f64, sigma: f64, nu: f64) -> f64 {
    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len() as f64;
    sorted
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let empirical = (i + 1) as f64 / n;
            let theoretical = t_cdf_std((x - mu) / sigma, nu);
            (empirical - theoretical).abs()
        })
        .fold(0.0f64, f64::max)
}

/// Lentz continued-fraction for regularised incomplete beta I_x(a,b).
fn regularised_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let lbeta_ab = lgamma(a + b) - lgamma(a) - lgamma(b);
    let front = x.powf(a) * (1.0 - x).powf(b) * lbeta_ab.exp() / a;
    front * betacf(a, b, x)
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let eps = 3.0e-7;
    let mut c = 1.0f64;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    d = if d.abs() < f64::MIN_POSITIVE { f64::MIN_POSITIVE } else { 1.0 / d };
    let mut h = d;
    for m in 1usize..=200 {
        let mf = m as f64;
        let aa = mf * (b - mf) * x / ((a + 2.0 * mf - 1.0) * (a + 2.0 * mf));
        d = 1.0 + aa * d;
        d = if d.abs() < f64::MIN_POSITIVE { f64::MIN_POSITIVE } else { d };
        c = 1.0 + aa / c;
        c = if c.abs() < f64::MIN_POSITIVE { f64::MIN_POSITIVE } else { c };
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + mf) * (a + b + mf) * x / ((a + 2.0 * mf) * (a + 2.0 * mf + 1.0));
        d = 1.0 + aa * d;
        d = if d.abs() < f64::MIN_POSITIVE { f64::MIN_POSITIVE } else { d };
        c = 1.0 + aa / c;
        c = if c.abs() < f64::MIN_POSITIVE { f64::MIN_POSITIVE } else { c };
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < eps {
            break;
        }
    }
    h
}

// ── Cauchy internals ─────────────────────────────────────────────────────────

fn cauchy_log_pdf(x: f64, x0: f64, gamma: f64) -> f64 {
    -std::f64::consts::PI.ln() - gamma.ln() - (1.0 + ((x - x0) / gamma).powi(2)).ln()
}

fn cauchy_log_likelihood(returns: &[f64], x0: f64, gamma: f64) -> f64 {
    returns.iter().map(|&r| cauchy_log_pdf(r, x0, gamma)).sum()
}

fn cauchy_ks_statistic(returns: &[f64], x0: f64, gamma: f64) -> f64 {
    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len() as f64;
    sorted
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let empirical = (i + 1) as f64 / n;
            let theoretical =
                0.5 + ((x - x0) / gamma).atan() / std::f64::consts::PI;
            (empirical - theoretical).abs()
        })
        .fold(0.0f64, f64::max)
}

// ── Discrete internals ───────────────────────────────────────────────────────

fn discrete_cdf(edges: &[f64], probs: &[f64], bin_width: f64, x: f64) -> f64 {
    let lo = edges[0];
    let hi = *edges.last().unwrap();
    if x <= lo {
        return 0.0;
    }
    if x >= hi {
        return 1.0;
    }
    let n_bins = probs.len();
    let idx = ((x - lo) / bin_width) as usize;
    let idx = idx.min(n_bins - 1);
    let complete: f64 = probs[..idx].iter().sum();
    let frac = ((x - edges[idx]) / bin_width).clamp(0.0, 1.0);
    complete + probs[idx] * frac
}
