//! Multivariate Hawkes process for order arrival modelling.
//!
//! Model:
//!   λ_buy(t)  = μ_buy  + α_bb · Σ_{buy  arrivals} exp(-β·(t-t_i))
//!                       + α_sb · Σ_{sell arrivals} exp(-β·(t-t_i))
//!   λ_sell(t) = μ_sell + α_ss · Σ_{sell arrivals} exp(-β·(t-t_i))
//!                       + α_bs · Σ_{buy  arrivals} exp(-β·(t-t_i))
//!
//! Parameters {μ_buy, μ_sell, α_bb, α_ss, α_bs, α_sb, β} are estimated by
//! MLE using a gradient ascent on the conditional log-likelihood.

use anyhow::{bail, Result};

/// Fitted Hawkes process parameters for one asset.
#[derive(Debug, Clone)]
pub struct HawkesProcess {
    pub mu_buy: f64,
    pub mu_sell: f64,
    /// Self-excitation: buy → buy.
    pub alpha_bb: f64,
    /// Self-excitation: sell → sell.
    pub alpha_ss: f64,
    /// Cross-excitation: buy → sell.
    pub alpha_bs: f64,
    /// Cross-excitation: sell → buy.
    pub alpha_sb: f64,
    pub beta: f64,
    pub log_likelihood: f64,
}

impl HawkesProcess {
    /// Estimate Hawkes parameters from trade arrival times (milliseconds).
    ///
    /// `buy_times` and `sell_times` must be sorted ascending.
    pub fn fit(buy_times: &[u64], sell_times: &[u64]) -> Result<Self> {
        if buy_times.len() < 10 || sell_times.len() < 10 {
            bail!(
                "Need ≥ 10 events per side (have buy={}, sell={})",
                buy_times.len(),
                sell_times.len()
            );
        }

        // Convert ms to seconds.
        let t_buy: Vec<f64> = buy_times.iter().map(|&t| t as f64 / 1000.0).collect();
        let t_sell: Vec<f64> = sell_times.iter().map(|&t| t as f64 / 1000.0).collect();
        let t_end = t_buy.last().copied().unwrap_or(0.0).max(t_sell.last().copied().unwrap_or(0.0));

        // Initialise at baseline Poisson rates.
        let mut mu_buy = t_buy.len() as f64 / t_end.max(1.0);
        let mut mu_sell = t_sell.len() as f64 / t_end.max(1.0);
        let mut alpha_bb = 0.3f64;
        let mut alpha_ss = 0.3f64;
        let mut alpha_bs = 0.1f64;
        let mut alpha_sb = 0.1f64;
        let mut beta = 1.0f64;

        let lr = 1e-5;
        let mut best_ll = f64::NEG_INFINITY;
        let mut best_params = (mu_buy, mu_sell, alpha_bb, alpha_ss, alpha_bs, alpha_sb, beta);

        for _ in 0..3000 {
            let ll = hawkes_ll(
                &t_buy, &t_sell, t_end, mu_buy, mu_sell, alpha_bb, alpha_ss, alpha_bs, alpha_sb,
                beta,
            );
            if ll > best_ll {
                best_ll = ll;
                best_params = (mu_buy, mu_sell, alpha_bb, alpha_ss, alpha_bs, alpha_sb, beta);
            }

            let eps = 1e-6;
            let params = [mu_buy, mu_sell, alpha_bb, alpha_ss, alpha_bs, alpha_sb, beta];
            let grads: Vec<f64> = (0..7usize)
                .map(|i| {
                    let mut p_hi = params;
                    let mut p_lo = params;
                    p_hi[i] += eps;
                    p_lo[i] -= eps;
                    (hawkes_ll_vec(&t_buy, &t_sell, t_end, &p_hi)
                        - hawkes_ll_vec(&t_buy, &t_sell, t_end, &p_lo))
                        / (2.0 * eps)
                })
                .collect();

            mu_buy = (mu_buy + lr * grads[0]).max(1e-8);
            mu_sell = (mu_sell + lr * grads[1]).max(1e-8);
            alpha_bb = (alpha_bb + lr * grads[2]).max(0.0);
            alpha_ss = (alpha_ss + lr * grads[3]).max(0.0);
            alpha_bs = (alpha_bs + lr * grads[4]).max(0.0);
            alpha_sb = (alpha_sb + lr * grads[5]).max(0.0);
            beta = (beta + lr * grads[6]).max(0.01);
        }

        (mu_buy, mu_sell, alpha_bb, alpha_ss, alpha_bs, alpha_sb, beta) = best_params;

        Ok(Self {
            mu_buy,
            mu_sell,
            alpha_bb,
            alpha_ss,
            alpha_bs,
            alpha_sb,
            beta,
            log_likelihood: best_ll,
        })
    }

    /// Cross-excitation ratio: how strongly a buy triggers further sells.
    /// High values indicate informed trading or stop-loss cascades.
    pub fn cross_excitation_ratio(&self) -> f64 {
        if self.alpha_bb < 1e-10 {
            return 0.0;
        }
        self.alpha_bs / self.alpha_bb
    }

    /// Order flow imbalance implied by background intensities.
    pub fn intensity_ofi(&self) -> f64 {
        let total = self.mu_buy + self.mu_sell;
        if total < 1e-10 {
            return 0.0;
        }
        (self.mu_buy - self.mu_sell) / total
    }
}

// ─── private ────────────────────────────────────────────────────────────────

fn hawkes_ll_vec(t_buy: &[f64], t_sell: &[f64], t_end: f64, p: &[f64]) -> f64 {
    hawkes_ll(t_buy, t_sell, t_end, p[0], p[1], p[2], p[3], p[4], p[5], p[6])
}

#[allow(clippy::too_many_arguments)]
fn hawkes_ll(
    t_buy: &[f64],
    t_sell: &[f64],
    t_end: f64,
    mu_buy: f64,
    mu_sell: f64,
    alpha_bb: f64,
    alpha_ss: f64,
    alpha_bs: f64,
    alpha_sb: f64,
    beta: f64,
) -> f64 {
    if mu_buy <= 0.0
        || mu_sell <= 0.0
        || alpha_bb < 0.0
        || alpha_ss < 0.0
        || alpha_bs < 0.0
        || alpha_sb < 0.0
        || beta <= 0.0
    {
        return f64::NEG_INFINITY;
    }

    // Merge and sort all events.
    let mut events: Vec<(f64, bool)> = t_buy
        .iter()
        .map(|&t| (t, true))
        .chain(t_sell.iter().map(|&t| (t, false)))
        .collect();
    events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut ll = 0.0f64;
    let mut r_buy = 0.0f64; // Σ exp(-β(t - t_i)) for buy events before current
    let mut r_sell = 0.0f64;

    for (i, &(t, is_buy)) in events.iter().enumerate() {
        // Decay previous sums.
        let prev_t = if i == 0 { 0.0 } else { events[i - 1].0 };
        let dt = t - prev_t;
        let decay = (-beta * dt).exp();
        r_buy *= decay;
        r_sell *= decay;

        // Intensity at arrival time.
        let lambda = if is_buy {
            mu_buy + alpha_bb * r_buy + alpha_sb * r_sell
        } else {
            mu_sell + alpha_ss * r_sell + alpha_bs * r_buy
        };
        if lambda <= 0.0 {
            return f64::NEG_INFINITY;
        }
        ll += lambda.ln();

        // Add this event to the appropriate sum.
        if is_buy {
            r_buy += 1.0;
        } else {
            r_sell += 1.0;
        }
    }

    // Compensator integral.
    let integral_buy = mu_buy * t_end
        + (alpha_bb / beta) * t_buy.iter().map(|&ti| 1.0 - (-beta * (t_end - ti)).exp()).sum::<f64>()
        + (alpha_sb / beta) * t_sell.iter().map(|&ti| 1.0 - (-beta * (t_end - ti)).exp()).sum::<f64>();
    let integral_sell = mu_sell * t_end
        + (alpha_ss / beta) * t_sell.iter().map(|&ti| 1.0 - (-beta * (t_end - ti)).exp()).sum::<f64>()
        + (alpha_bs / beta) * t_buy.iter().map(|&ti| 1.0 - (-beta * (t_end - ti)).exp()).sum::<f64>();

    ll - integral_buy - integral_sell
}
