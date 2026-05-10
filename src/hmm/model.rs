//! Hidden Markov Model with Gaussian emissions and Baum–Welch training.
//!
//! Four states:
//!   S0 – Trending Bull  (rising prices, positive flow, low spread)
//!   S1 – Trending Bear  (falling prices, negative flow, widening spread)
//!   S2 – Consolidation  (range-bound, low vol, mean-reversion regime)
//!   S3 – High-Vol Chaos (extreme vol, erratic flow, wide spreads)
//!
//! Observation vector (7-dim):
//!   [directional_edge_btc, directional_edge_eth, predicted_vol_agg,
//!    cross_asset_correlation, portfolio_gross_exposure,
//!    funding_z_mean, n_open_positions_norm]

use anyhow::{bail, Result};
use nalgebra::{DMatrix, DVector};

/// Market regime label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regime {
    TrendingBull = 0,
    TrendingBear = 1,
    Consolidation = 2,
    HighVol = 3,
}

impl Regime {
    pub fn from_index(i: usize) -> Self {
        match i {
            0 => Self::TrendingBull,
            1 => Self::TrendingBear,
            2 => Self::Consolidation,
            _ => Self::HighVol,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Regime::TrendingBull => "Trending Bull",
            Regime::TrendingBear => "Trending Bear",
            Regime::Consolidation => "Consolidation",
            Regime::HighVol => "High-Vol Chaos",
        }
    }

    /// Maximum leverage permitted in this regime.
    pub fn max_leverage(&self) -> f64 {
        match self {
            Regime::TrendingBull | Regime::TrendingBear => 5.0,
            Regime::Consolidation => 2.0,
            Regime::HighVol => 1.0,
        }
    }

    /// Position size multiplier relative to a full position.
    pub fn size_scale(&self) -> f64 {
        match self {
            Regime::TrendingBull | Regime::TrendingBear => 1.0,
            Regime::Consolidation => 0.5,
            Regime::HighVol => 0.2,
        }
    }

    /// Minimum risk-reward ratio for signals in this regime.
    pub fn min_rr(&self) -> f64 {
        match self {
            Regime::TrendingBull | Regime::TrendingBear => 2.0,
            Regime::Consolidation => 1.5,
            Regime::HighVol => 3.0,
        }
    }
}

/// Current regime state with uncertainty estimate.
#[derive(Debug, Clone)]
pub struct RegimeState {
    pub regime: Regime,
    pub probability: f64,
    pub posterior: Vec<f64>, // P(S=i | observations) for i=0..3
}

impl RegimeState {
    pub fn is_confident(&self) -> bool {
        self.probability > 0.65
    }
}

/// HMM parameters.
pub struct HmmModel {
    n_states: usize,
    n_obs: usize,
    /// Initial state distribution π.
    pub pi: DVector<f64>,
    /// Transition matrix A (n_states × n_states): A[i,j] = P(S_t=j | S_{t-1}=i).
    pub transition: DMatrix<f64>,
    /// Emission means μ (n_states × n_obs).
    pub means: DMatrix<f64>,
    /// Emission diagonal covariances Σ (n_states × n_obs) — diagonal Gaussian.
    pub covariances: DMatrix<f64>,
}

impl HmmModel {
    /// Create an HMM with sensible priors for the four crypto regimes.
    pub fn new(n_obs: usize) -> Self {
        let n = 4usize;
        // Transition matrix: high self-persistence, small transition probabilities.
        let t = DMatrix::from_row_slice(
            n,
            n,
            &[
                // S0 Bull  S1 Bear  S2 Consol  S3 HiVol
                0.92, 0.04, 0.03, 0.01, // from Bull
                0.05, 0.89, 0.03, 0.03, // from Bear
                0.06, 0.04, 0.87, 0.03, // from Consol
                0.15, 0.15, 0.08, 0.62, // from HiVol
            ],
        );

        // Emission means: [dir_edge_btc, dir_edge_eth, vol_agg, corr, exposure, fund_z, n_pos]
        let means = DMatrix::from_row_slice(
            n,
            n_obs,
            &[
                // Bull: positive edges, moderate vol, high exposure
                0.35, 0.30, 0.012, 0.80, 0.70, -0.5, 0.60,
                // Bear: negative edges, rising vol, short exposure
                -0.35, -0.30, 0.018, 0.75, 0.60, 0.5, 0.50,
                // Consol: near-zero edges, low vol, low exposure
                0.05, 0.05, 0.006, 0.60, 0.30, 0.0, 0.30,
                // HiVol: erratic edges, extreme vol, low exposure
                0.0, 0.0, 0.035, 0.40, 0.20, 1.0, 0.20,
            ],
        );

        // Emission covariances (diagonal): how spread each state's observations are.
        let covariances = DMatrix::from_row_slice(
            n,
            n_obs,
            &[
                0.04, 0.04, 0.0001, 0.02, 0.02, 0.25, 0.02,
                0.04, 0.04, 0.0002, 0.02, 0.02, 0.25, 0.02,
                0.01, 0.01, 0.00005, 0.04, 0.01, 0.10, 0.01,
                0.10, 0.10, 0.0010, 0.06, 0.01, 0.50, 0.01,
            ],
        );

        Self {
            n_states: n,
            n_obs,
            pi: DVector::from_vec(vec![0.25, 0.25, 0.25, 0.25]),
            transition: t,
            means,
            covariances,
        }
    }

    /// Train using Baum–Welch EM on a sequence of observations.
    ///
    /// Each observation is a `n_obs`-dimensional vector.
    /// Convergence is declared when log-likelihood improvement < 1e-6.
    pub fn fit(&mut self, observations: &[Vec<f64>]) -> Result<()> {
        let t_len = observations.len();
        if t_len < 10 {
            bail!("Need ≥ 10 observation steps to train HMM");
        }

        let max_iter = 200;
        let mut prev_ll = f64::NEG_INFINITY;

        for _iter in 0..max_iter {
            // ── Forward pass ────────────────────────────────────────────────
            let (alpha, ll) = self.forward(observations);
            if ll < f64::NEG_INFINITY + 1.0 {
                bail!("HMM forward pass produced -inf; check observations");
            }

            if (ll - prev_ll).abs() < 1e-6 {
                break;
            }
            prev_ll = ll;

            // ── Backward pass ────────────────────────────────────────────────
            let beta = self.backward(observations);

            // ── E-step: γ and ξ ─────────────────────────────────────────────
            let gamma = self.gamma(&alpha, &beta);
            let xi = self.xi(observations, &alpha, &beta);

            // ── M-step ──────────────────────────────────────────────────────
            // Update π.
            for i in 0..self.n_states {
                self.pi[i] = gamma[0][i];
            }

            // Update A.
            for i in 0..self.n_states {
                let row_sum: f64 = (0..t_len - 1)
                    .map(|t| xi[t].row(i).sum())
                    .sum::<f64>();
                for j in 0..self.n_states {
                    let num: f64 = (0..t_len - 1).map(|t| xi[t][(i, j)]).sum();
                    self.transition[(i, j)] = num / row_sum.max(1e-300);
                }
                // Renormalise row.
                let row_sum2: f64 = (0..self.n_states).map(|j| self.transition[(i, j)]).sum();
                for j in 0..self.n_states {
                    self.transition[(i, j)] /= row_sum2.max(1e-300);
                }
            }

            // Update means and covariances.
            for i in 0..self.n_states {
                let g_sum: f64 = gamma.iter().map(|gt| gt[i]).sum::<f64>().max(1e-300);
                for k in 0..self.n_obs {
                    let mu_new: f64 = gamma
                        .iter()
                        .zip(observations)
                        .map(|(gt, ot)| gt[i] * ot[k])
                        .sum::<f64>()
                        / g_sum;
                    let var_new: f64 = gamma
                        .iter()
                        .zip(observations)
                        .map(|(gt, ot)| gt[i] * (ot[k] - mu_new).powi(2))
                        .sum::<f64>()
                        / g_sum;
                    self.means[(i, k)] = mu_new;
                    self.covariances[(i, k)] = var_new.max(1e-10);
                }
            }
        }

        Ok(())
    }

    /// Viterbi decoding — returns most-likely state sequence.
    pub fn viterbi(&self, observations: &[Vec<f64>]) -> Vec<usize> {
        let t_len = observations.len();
        if t_len == 0 {
            return vec![];
        }

        let mut viterbi = vec![vec![0.0f64; self.n_states]; t_len];
        let mut backptr = vec![vec![0usize; self.n_states]; t_len];

        // Initialise.
        for i in 0..self.n_states {
            viterbi[0][i] = self.pi[i].ln() + self.log_emission(i, &observations[0]);
        }

        // Recursion.
        for t in 1..t_len {
            for j in 0..self.n_states {
                let (best_i, best_v) = (0..self.n_states)
                    .map(|i| {
                        let v = viterbi[t - 1][i] + self.transition[(i, j)].ln();
                        (i, v)
                    })
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap();
                viterbi[t][j] = best_v + self.log_emission(j, &observations[t]);
                backptr[t][j] = best_i;
            }
        }

        // Backtrack.
        let mut path = vec![0usize; t_len];
        path[t_len - 1] = viterbi[t_len - 1]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        for t in (0..t_len - 1).rev() {
            path[t] = backptr[t + 1][path[t + 1]];
        }
        path
    }

    /// Forward algorithm — returns (α, log-likelihood).
    pub fn forward(&self, observations: &[Vec<f64>]) -> (Vec<Vec<f64>>, f64) {
        let t_len = observations.len();
        let mut alpha = vec![vec![0.0f64; self.n_states]; t_len];
        let mut log_scale = 0.0f64;

        // Initialise.
        for i in 0..self.n_states {
            alpha[0][i] = self.pi[i] * self.emission(i, &observations[0]);
        }
        let scale = alpha[0].iter().sum::<f64>().max(1e-300);
        for a in alpha[0].iter_mut() {
            *a /= scale;
        }
        log_scale += scale.ln();

        // Recursion.
        for t in 1..t_len {
            for j in 0..self.n_states {
                alpha[t][j] = (0..self.n_states)
                    .map(|i| alpha[t - 1][i] * self.transition[(i, j)])
                    .sum::<f64>()
                    * self.emission(j, &observations[t]);
            }
            let scale = alpha[t].iter().sum::<f64>().max(1e-300);
            for a in alpha[t].iter_mut() {
                *a /= scale;
            }
            log_scale += scale.ln();
        }

        (alpha, log_scale)
    }

    /// Compute posterior state probabilities at the last timestep (online filtering).
    pub fn filter_last(&self, observations: &[Vec<f64>]) -> RegimeState {
        let (alpha, _) = self.forward(observations);
        let last = &alpha[alpha.len() - 1];
        let total: f64 = last.iter().sum::<f64>().max(1e-300);
        let posterior: Vec<f64> = last.iter().map(|a| a / total).collect();
        let (best_i, best_p) = posterior
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        RegimeState {
            regime: Regime::from_index(best_i),
            probability: *best_p,
            posterior,
        }
    }

    // ─── private ────────────────────────────────────────────────────────────

    fn backward(&self, observations: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let t_len = observations.len();
        let mut beta = vec![vec![1.0f64; self.n_states]; t_len];

        for t in (0..t_len - 1).rev() {
            for i in 0..self.n_states {
                beta[t][i] = (0..self.n_states)
                    .map(|j| {
                        self.transition[(i, j)]
                            * self.emission(j, &observations[t + 1])
                            * beta[t + 1][j]
                    })
                    .sum::<f64>();
            }
            let scale = beta[t].iter().sum::<f64>().max(1e-300);
            for b in beta[t].iter_mut() {
                *b /= scale;
            }
        }
        beta
    }

    fn gamma(&self, alpha: &[Vec<f64>], beta: &[Vec<f64>]) -> Vec<Vec<f64>> {
        alpha
            .iter()
            .zip(beta)
            .map(|(a, b)| {
                let raw: Vec<f64> = a.iter().zip(b).map(|(&ai, &bi)| ai * bi).collect();
                let s = raw.iter().sum::<f64>().max(1e-300);
                raw.iter().map(|&x| x / s).collect()
            })
            .collect()
    }

    fn xi(
        &self,
        observations: &[Vec<f64>],
        alpha: &[Vec<f64>],
        beta: &[Vec<f64>],
    ) -> Vec<DMatrix<f64>> {
        let t_len = observations.len();
        let mut xi = Vec::with_capacity(t_len - 1);
        for t in 0..t_len - 1 {
            let mut m = DMatrix::zeros(self.n_states, self.n_states);
            for i in 0..self.n_states {
                for j in 0..self.n_states {
                    m[(i, j)] = alpha[t][i]
                        * self.transition[(i, j)]
                        * self.emission(j, &observations[t + 1])
                        * beta[t + 1][j];
                }
            }
            let s = m.sum().max(1e-300);
            xi.push(m / s);
        }
        xi
    }

    /// Diagonal Gaussian emission probability.
    fn emission(&self, state: usize, obs: &[f64]) -> f64 {
        let log_p = self.log_emission(state, obs);
        log_p.exp().max(1e-300)
    }

    fn log_emission(&self, state: usize, obs: &[f64]) -> f64 {
        let mut log_p = 0.0f64;
        for k in 0..self.n_obs.min(obs.len()) {
            let mu = self.means[(state, k)];
            let var = self.covariances[(state, k)];
            let diff = obs[k] - mu;
            log_p += -0.5 * (diff * diff / var + var.ln() + std::f64::consts::TAU.ln());
        }
        log_p
    }
}
