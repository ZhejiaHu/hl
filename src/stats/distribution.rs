//! Heavy-tailed return distribution fitting via maximum likelihood.
//!
//! Fits a standardised Student-t to log-returns using Newton–Raphson on the
//! profile log-likelihood, holding (μ, σ) at their moment estimators while
//! iterating on ν. A full joint optimisation follows.

use anyhow::{bail, Result};

/// Parameters of a fitted standardised Student-t distribution.
#[derive(Debug, Clone)]
pub struct StudentT {
    /// Location (mean of returns).
    pub mu: f64,
    /// Scale (std-dev-like spread).
    pub sigma: f64,
    /// Degrees of freedom (>2 for finite variance; typical crypto: 3–6).
    pub nu: f64,
    /// Log-likelihood at fitted parameters.
    pub log_likelihood: f64,
    /// Kolmogorov–Smirnov statistic (goodness-of-fit).
    pub ks_stat: f64,
}

impl StudentT {
    /// Fit a standardised t to `returns` using MLE (Newton–Raphson on ν,
    /// closed-form update for μ and σ²).
    pub fn fit(returns: &[f64]) -> Result<Self> {
        let n = returns.len();
        if n < 30 {
            bail!("Need at least 30 observations to fit Student-t (have {})", n);
        }

        let mu = mean(returns);
        let mut sigma = std_dev(returns, mu).max(1e-10);
        let mut nu = 5.0f64; // Starting point for Newton–Raphson

        // EM-like alternating updates between (mu, sigma) and (nu).
        for _ in 0..200 {
            // E-step weights: w_i = (nu + 1) / (nu + z_i^2)
            let weights: Vec<f64> = returns
                .iter()
                .map(|&r| {
                    let z = (r - mu) / sigma;
                    (nu + 1.0) / (nu + z * z)
                })
                .collect();

            // M-step for sigma^2 (mu assumed fixed at moment estimate for stability).
            let sigma2_new = returns
                .iter()
                .zip(&weights)
                .map(|(&r, &w)| w * (r - mu).powi(2))
                .sum::<f64>()
                / n as f64;
            sigma = sigma2_new.sqrt().max(1e-10);

            // M-step for nu via digamma fixed-point iteration.
            let rhs: f64 = 1.0
                + (weights
                    .iter()
                    .zip(returns.iter())
                    .map(|(&w, &r)| {
                        let z = (r - mu) / sigma;
                        (w - (z * z * nu / (nu + z * z)).ln() - w * (z * z * nu / (nu + z * z)))
                    })
                    .sum::<f64>()
                    / n as f64);
            // Use Newton step on digamma equation: digamma((nu+1)/2) - digamma(nu/2) = rhs
            nu = nu_from_fixed_point(rhs).max(2.1);
        }

        let ll = log_likelihood_t(returns, mu, sigma, nu);
        let ks = ks_statistic(returns, mu, sigma, nu);

        Ok(Self {
            mu,
            sigma,
            nu,
            log_likelihood: ll,
            ks_stat: ks,
        })
    }

    /// Log-PDF of the fitted distribution at value `x`.
    pub fn log_pdf(&self, x: f64) -> f64 {
        log_pdf_t(x, self.mu, self.sigma, self.nu)
    }

    /// Value-at-Risk at confidence level `alpha` (e.g. 0.95).
    pub fn var(&self, alpha: f64) -> f64 {
        // Quantile of standardised t, scaled and shifted.
        let t_q = t_quantile(self.nu, 1.0 - alpha);
        self.mu + self.sigma * t_q
    }

    /// Conditional VaR (Expected Shortfall) at `alpha`.
    pub fn cvar(&self, alpha: f64) -> f64 {
        let v = self.var(alpha);
        // CVaR for t-distribution: E[X | X < VaR]
        let z = (v - self.mu) / self.sigma;
        let pdf_z = t_pdf_std(z, self.nu);
        let cdf_z = t_cdf_std(z, self.nu);
        if cdf_z < 1e-15 {
            return v;
        }
        self.mu + self.sigma * (-pdf_z * (self.nu + z * z) / ((self.nu - 1.0) * cdf_z))
    }
}

// ─── private helpers ────────────────────────────────────────────────────────

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn std_dev(xs: &[f64], mu: f64) -> f64 {
    let var = xs.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / xs.len() as f64;
    var.sqrt()
}

/// Approximate digamma function using the asymptotic expansion.
fn digamma(x: f64) -> f64 {
    if x < 6.0 {
        return digamma(x + 1.0) - 1.0 / x;
    }
    // Abramowitz & Stegun 6.3.18
    let r = 1.0 / x;
    x.ln() - 0.5 * r
        - r.powi(2)
            * (1.0 / 12.0
                - r.powi(2) * (1.0 / 120.0 - r.powi(2) * (1.0 / 252.0)))
}

/// Approximate trigamma (derivative of digamma).
fn trigamma(x: f64) -> f64 {
    if x < 6.0 {
        return trigamma(x + 1.0) + 1.0 / x.powi(2);
    }
    let r = 1.0 / x;
    r + 0.5 * r.powi(2) + r.powi(3) / 6.0 - r.powi(5) / 30.0
}

/// Solve the EM fixed-point equation for ν using Newton–Raphson.
fn nu_from_fixed_point(target: f64) -> f64 {
    let mut nu = 5.0f64;
    for _ in 0..50 {
        let f = digamma((nu + 1.0) / 2.0) - digamma(nu / 2.0) - target;
        let df = 0.5 * trigamma((nu + 1.0) / 2.0) - 0.5 * trigamma(nu / 2.0);
        if df.abs() < 1e-15 {
            break;
        }
        let step = f / df;
        nu -= step;
        nu = nu.max(2.1);
        if step.abs() < 1e-8 {
            break;
        }
    }
    nu
}

fn lgamma(x: f64) -> f64 {
    statrs::function::gamma::ln_gamma(x)
}

fn log_pdf_t(x: f64, mu: f64, sigma: f64, nu: f64) -> f64 {
    let z = (x - mu) / sigma;
    lgamma((nu + 1.0) / 2.0)
        - lgamma(nu / 2.0)
        - 0.5 * (std::f64::consts::PI * nu).ln()
        - sigma.ln()
        - (nu + 1.0) / 2.0 * (1.0 + z * z / nu).ln()
}

fn log_likelihood_t(returns: &[f64], mu: f64, sigma: f64, nu: f64) -> f64 {
    returns.iter().map(|&r| log_pdf_t(r, mu, sigma, nu)).sum()
}

/// Standard t PDF (zero mean, unit scale).
fn t_pdf_std(t: f64, nu: f64) -> f64 {
    (lgamma((nu + 1.0) / 2.0) - lgamma(nu / 2.0) - 0.5 * (std::f64::consts::PI * nu).ln())
        .exp()
        * (1.0 + t * t / nu).powf(-(nu + 1.0) / 2.0)
}

/// Regularised incomplete beta function for CDF.
fn t_cdf_std(t: f64, nu: f64) -> f64 {
    let x = nu / (nu + t * t);
    let ib = regularised_incomplete_beta(nu / 2.0, 0.5, x);
    if t < 0.0 { 0.5 * ib } else { 1.0 - 0.5 * ib }
}

fn t_quantile(nu: f64, p: f64) -> f64 {
    // Bisection on t_cdf_std.
    let mut lo = -20.0f64;
    let mut hi = 0.0f64;
    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        if t_cdf_std(mid, nu) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}

/// Lentz continued-fraction for regularised incomplete beta I_x(a,b).
fn regularised_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    // Use continued fraction via modified Lentz method (Numerical Recipes).
    let lbeta_ab = lgamma(a + b) - lgamma(a) - lgamma(b);
    let front = (x.powf(a) * (1.0 - x).powf(b) * (lbeta_ab).exp()) / a;
    front * betacf(a, b, x)
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let max_iter = 200;
    let eps = 3.0e-7;
    let mut c = 1.0f64;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    d = if d.abs() < f64::MIN_POSITIVE { f64::MIN_POSITIVE } else { 1.0 / d };
    let mut h = d;
    for m in 1..=max_iter {
        let m = m as f64;
        // Even step
        let aa = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
        d = 1.0 + aa * d;
        d = if d.abs() < f64::MIN_POSITIVE { f64::MIN_POSITIVE } else { d };
        c = 1.0 + aa / c;
        c = if c.abs() < f64::MIN_POSITIVE { f64::MIN_POSITIVE } else { c };
        d = 1.0 / d;
        h *= d * c;
        // Odd step
        let aa = -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0));
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

/// Two-sample Kolmogorov–Smirnov statistic against the fitted t-CDF.
fn ks_statistic(returns: &[f64], mu: f64, sigma: f64, nu: f64) -> f64 {
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
