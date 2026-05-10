//! GARCH(1,1) model with Student-t innovations.
//!
//! σ²_t = ω + α · ε²_{t-1} + β · σ²_{t-1}
//!
//! Parameters are estimated by maximum likelihood using projected gradient
//! descent with the stationarity constraint α + β < 1.

use anyhow::{bail, Result};

/// Fitted GARCH(1,1) parameters.
#[derive(Debug, Clone)]
pub struct Garch11 {
    /// Constant term (unconditional variance baseline).
    pub omega: f64,
    /// ARCH coefficient — sensitivity to past squared shocks.
    pub alpha: f64,
    /// GARCH coefficient — persistence of conditional variance.
    pub beta: f64,
    /// Degrees of freedom for t-distributed innovations.
    pub nu: f64,
    /// Log-likelihood at convergence.
    pub log_likelihood: f64,
}

impl Garch11 {
    /// Fit GARCH(1,1)-t to the return series using MLE.
    pub fn fit(returns: &[f64]) -> Result<Self> {
        let n = returns.len();
        if n < 50 {
            bail!("Need at least 50 observations for GARCH (have {})", n);
        }

        // Initialise parameters from moment estimators.
        let var_unc = sample_variance(returns);
        let mut omega = var_unc * 0.05;
        let mut alpha = 0.10f64;
        let mut beta = 0.85f64;
        let mut nu = 6.0f64;
        let lr = 1e-5;

        // Gradient descent with numerical gradients (central differences).
        let mut best_ll = f64::NEG_INFINITY;
        let mut best = (omega, alpha, beta, nu);

        for _iter in 0..5000 {
            let ll = garch_ll(returns, omega, alpha, beta, nu);
            if ll > best_ll {
                best_ll = ll;
                best = (omega, alpha, beta, nu);
            }

            let eps = 1e-7;
            let d_omega = (garch_ll(returns, omega + eps, alpha, beta, nu)
                - garch_ll(returns, omega - eps, alpha, beta, nu))
                / (2.0 * eps);
            let d_alpha = (garch_ll(returns, omega, alpha + eps, beta, nu)
                - garch_ll(returns, omega, alpha - eps, beta, nu))
                / (2.0 * eps);
            let d_beta = (garch_ll(returns, omega, alpha, beta + eps, nu)
                - garch_ll(returns, omega, alpha, beta - eps, nu))
                / (2.0 * eps);
            let d_nu = (garch_ll(returns, omega, alpha, beta, nu + eps)
                - garch_ll(returns, omega, alpha, beta, nu - eps))
                / (2.0 * eps);

            omega = (omega + lr * d_omega).max(1e-12);
            alpha = (alpha + lr * d_alpha).clamp(0.01, 0.49);
            beta = (beta + lr * d_beta).clamp(0.01, 0.98);
            nu = (nu + lr * 100.0 * d_nu).max(2.1);

            // Project onto stationarity manifold α + β < 1.
            if alpha + beta >= 0.9999 {
                let scale = 0.9999 / (alpha + beta);
                alpha *= scale;
                beta *= scale;
            }
        }

        (omega, alpha, beta, nu) = best;

        Ok(Self {
            omega,
            alpha,
            beta,
            nu,
            log_likelihood: best_ll,
        })
    }

    /// Compute the full conditional variance sequence.
    pub fn conditional_variances(&self, returns: &[f64]) -> Vec<f64> {
        conditional_variances_inner(returns, self.omega, self.alpha, self.beta)
    }

    /// One-step-ahead conditional standard deviation.
    pub fn forecast_sigma(&self, returns: &[f64]) -> f64 {
        let variances = self.conditional_variances(returns);
        let last_var = variances.last().copied().unwrap_or(self.unconditional_var());
        let last_eps2 = returns.last().map(|r| r * r).unwrap_or(0.0);
        let forecast_var = self.omega + self.alpha * last_eps2 + self.beta * last_var;
        forecast_var.sqrt()
    }

    /// Long-run (unconditional) variance: ω / (1 - α - β).
    pub fn unconditional_var(&self) -> f64 {
        self.omega / (1.0 - self.alpha - self.beta).max(1e-10)
    }

    /// Annualised unconditional volatility.
    pub fn annual_vol(&self, bars_per_year: f64) -> f64 {
        (self.unconditional_var() * bars_per_year).sqrt()
    }
}

// ─── private ────────────────────────────────────────────────────────────────

fn sample_variance(xs: &[f64]) -> f64 {
    let mu = xs.iter().sum::<f64>() / xs.len() as f64;
    xs.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / xs.len() as f64
}

fn conditional_variances_inner(returns: &[f64], omega: f64, alpha: f64, beta: f64) -> Vec<f64> {
    let var_unc = (omega / (1.0 - alpha - beta).max(1e-10)).max(1e-12);
    let mut h = Vec::with_capacity(returns.len());
    let mut h_prev = var_unc;
    for &r in returns {
        let h_t = omega + alpha * r * r + beta * h_prev;
        h.push(h_t);
        h_prev = h_t;
    }
    h
}

fn lgamma(x: f64) -> f64 {
    statrs::function::gamma::ln_gamma(x)
}

/// GARCH(1,1)-t log-likelihood.
fn garch_ll(returns: &[f64], omega: f64, alpha: f64, beta: f64, nu: f64) -> f64 {
    if omega <= 0.0 || alpha <= 0.0 || beta <= 0.0 || alpha + beta >= 1.0 || nu <= 2.0 {
        return f64::NEG_INFINITY;
    }
    let variances = conditional_variances_inner(returns, omega, alpha, beta);
    let log_const = lgamma((nu + 1.0) / 2.0)
        - lgamma(nu / 2.0)
        - 0.5 * (std::f64::consts::PI * (nu - 2.0)).ln();
    returns
        .iter()
        .zip(&variances)
        .map(|(&r, &h)| {
            if h <= 0.0 {
                return f64::NEG_INFINITY;
            }
            log_const
                - 0.5 * h.ln()
                - (nu + 1.0) / 2.0 * (1.0 + r * r / (h * (nu - 2.0))).ln()
        })
        .sum()
}
