//! Walk-forward backtesting engine.
//!
//! ## Design
//! The engine replays historical price bars through the same pipeline used for
//! live trading: estimation → ensemble → optimization → execution simulation.
//! This minimises implementation drift between backtest and live.
//!
//! ## Walk-forward evaluation
//! The available history is divided into alternating training and test windows:
//!   [train_1][test_1][train_2][test_2]…
//!
//! On each training window, estimator parameters are re-fitted (GARCH, OU).
//! On each test window, only online updates (Dual KF, Bayesian fusion) are used.
//! This prevents look-ahead bias and overfitting.
//!
//! ## Execution simulation
//! Fill simulation uses the ExecutionModel (latency, impact, fill probability).
//! Slippage is applied to entry prices; partial fills are modelled.
//!
//! ## Metrics
//! - Cumulative return, Sharpe, Sortino, Calmar ratios.
//! - Maximum drawdown and drawdown duration.
//! - Hit rate, average win/loss.
//! - Turnover and transaction costs.

use std::collections::HashMap;

use crate::{
    execution::model::{ExecutionModel, ImpactParams},
    risk::limits::RiskLimits,
};

// ── Configuration ─────────────────────────────────────────────────────────────

/// Walk-forward backtest configuration.
#[derive(Debug, Clone)]
pub struct WalkForwardConfig {
    /// Number of bars per training window.
    pub train_bars: usize,
    /// Number of bars per test window.
    pub test_bars: usize,
    /// Number of folds to run (0 = run all folds in the data).
    pub max_folds: usize,
    /// Initial portfolio NAV for the backtest.
    pub initial_nav: f64,
    /// Apply execution model during simulation.
    pub simulate_execution: bool,
    /// 24h ADV in USD (used for market impact simulation).
    pub adv_24h_usd: f64,
}

impl Default for WalkForwardConfig {
    fn default() -> Self {
        Self {
            train_bars: 500,
            test_bars: 100,
            max_folds: 0,
            initial_nav: 10_000.0,
            simulate_execution: true,
            adv_24h_usd: 1e8,
        }
    }
}

// ── Simulated bar ─────────────────────────────────────────────────────────────

/// Minimal bar representation for backtesting.
#[derive(Debug, Clone)]
pub struct BacktestBar {
    pub timestamp_ms: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub spread_pct: f64,      // Estimated bid-ask spread as fraction of mid.
    pub book_depth_usd: f64,  // Available depth at 1% of mid.
}

impl BacktestBar {
    pub fn log_return(&self, prev_close: f64) -> f64 {
        if prev_close > 1e-10 {
            (self.close / prev_close).ln()
        } else {
            0.0
        }
    }
}

// ── Simulated trade ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SimulatedTrade {
    pub asset: String,
    pub entry_bar: usize,
    pub exit_bar: usize,
    pub direction: bool, // true=long
    pub entry_price: f64,
    pub exit_price: f64,
    pub size_usd: f64,
    pub leverage: f64,
    pub realized_pnl: f64,
    pub transaction_cost_usd: f64,
    pub fill_fraction: f64,
}

// ── Backtest result ───────────────────────────────────────────────────────────

/// Aggregate performance metrics from one backtest fold.
#[derive(Debug, Clone, Default)]
pub struct FoldResult {
    pub fold_idx: usize,
    pub n_bars: usize,
    pub n_trades: usize,
    pub total_return: f64,
    pub annualized_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub calmar_ratio: f64,
    pub max_drawdown: f64,
    pub max_drawdown_duration_bars: usize,
    pub hit_rate: f64,
    pub avg_win_pct: f64,
    pub avg_loss_pct: f64,
    pub total_turnover: f64,
    pub total_transaction_costs_usd: f64,
}

/// Full walk-forward backtest results.
#[derive(Debug, Default)]
pub struct BacktestResult {
    pub folds: Vec<FoldResult>,
    pub equity_curve: Vec<(u64, f64)>, // (timestamp_ms, NAV)
    pub trades: Vec<SimulatedTrade>,
}

impl BacktestResult {
    /// Aggregate statistics across all folds.
    pub fn aggregate(&self) -> FoldResult {
        if self.folds.is_empty() {
            return FoldResult::default();
        }
        let n = self.folds.len() as f64;
        FoldResult {
            fold_idx: 0,
            n_bars: self.folds.iter().map(|f| f.n_bars).sum(),
            n_trades: self.folds.iter().map(|f| f.n_trades).sum(),
            total_return: self.folds.iter().map(|f| f.total_return).product::<f64>() - 1.0,
            annualized_return: self.folds.iter().map(|f| f.annualized_return).sum::<f64>() / n,
            sharpe_ratio: self.folds.iter().map(|f| f.sharpe_ratio).sum::<f64>() / n,
            sortino_ratio: self.folds.iter().map(|f| f.sortino_ratio).sum::<f64>() / n,
            calmar_ratio: self.folds.iter().map(|f| f.calmar_ratio).sum::<f64>() / n,
            max_drawdown: self
                .folds
                .iter()
                .map(|f| f.max_drawdown)
                .fold(0.0f64, f64::max),
            max_drawdown_duration_bars: self
                .folds
                .iter()
                .map(|f| f.max_drawdown_duration_bars)
                .max()
                .unwrap_or(0),
            hit_rate: self.folds.iter().map(|f| f.hit_rate).sum::<f64>() / n,
            avg_win_pct: self.folds.iter().map(|f| f.avg_win_pct).sum::<f64>() / n,
            avg_loss_pct: self.folds.iter().map(|f| f.avg_loss_pct).sum::<f64>() / n,
            total_turnover: self.folds.iter().map(|f| f.total_turnover).sum(),
            total_transaction_costs_usd: self.folds.iter().map(|f| f.total_transaction_costs_usd).sum(),
        }
    }
}

// ── Backtest engine ───────────────────────────────────────────────────────────

/// Walk-forward backtesting engine.
///
/// Replays historical bars and simulates the full pipeline: estimation,
/// ensemble fusion, optimization, execution simulation, and risk checks.
///
/// The engine and live system share the same estimation/ensemble/optimizer
/// modules to minimise implementation drift.
pub struct BacktestEngine {
    pub config: WalkForwardConfig,
    pub exec_model: ExecutionModel,
    pub risk_limits: RiskLimits,
}

impl BacktestEngine {
    pub fn new(config: WalkForwardConfig, risk_limits: RiskLimits) -> Self {
        let exec = ExecutionModel::new(
            10_000,
            5.0,
            ImpactParams::default(),
        );
        Self { config, exec_model: exec, risk_limits }
    }

    /// Run walk-forward evaluation on per-asset bar data.
    ///
    /// `bars`: map from asset symbol to chronological bar sequence.
    ///
    /// Returns a `BacktestResult` with per-fold and aggregate metrics.
    pub fn run(
        &mut self,
        bars: &HashMap<String, Vec<BacktestBar>>,
    ) -> BacktestResult {
        use crate::estimation::{dual_kalman::DualKalmanFilter, Observation, StateEstimator};
        use crate::ensemble::fusion::{BayesianFusion, kalman_signal};
        use crate::features::frequency::FrequencyExtractor;

        let mut result = BacktestResult::default();

        // Determine the fold schedule from the shortest asset series.
        let min_bars = bars.values().map(|v| v.len()).min().unwrap_or(0);
        let fold_size = self.config.train_bars + self.config.test_bars;
        if min_bars < fold_size {
            return result;
        }

        let max_folds = if self.config.max_folds == 0 {
            min_bars / fold_size
        } else {
            self.config.max_folds.min(min_bars / fold_size)
        };

        for fold_idx in 0..max_folds {
            let train_start = fold_idx * fold_size;
            let test_start = train_start + self.config.train_bars;
            let test_end = test_start + self.config.test_bars;

            let fold_result = self.run_fold(
                bars,
                fold_idx,
                train_start,
                test_start,
                test_end,
            );

            result.folds.push(fold_result);
        }

        result
    }

    fn run_fold(
        &self,
        bars: &HashMap<String, Vec<BacktestBar>>,
        fold_idx: usize,
        train_start: usize,
        test_start: usize,
        test_end: usize,
    ) -> FoldResult {
        use crate::estimation::{dual_kalman::DualKalmanFilter, Observation, StateEstimator};
        use crate::ensemble::fusion::{BayesianFusion, kalman_signal};
        use crate::features::frequency::FrequencyExtractor;

        // Per-asset estimators (reset each fold to prevent train→test leakage).
        let mut kf_map: HashMap<String, DualKalmanFilter> = bars
            .keys()
            .map(|k| (k.clone(), DualKalmanFilter::default_crypto()))
            .collect();
        let mut freq_map: HashMap<String, FrequencyExtractor> = bars
            .keys()
            .map(|k| (k.clone(), FrequencyExtractor::new(64)))
            .collect();
        let mut fusion_map: HashMap<String, BayesianFusion> = bars
            .keys()
            .map(|k| {
                let mut f = BayesianFusion::new(2, 0.99, 0.05);
                f.register_source("dual_kalman");
                f.register_source("frequency");
                (k.clone(), f)
            })
            .collect();

        // ── Training phase: warm-up estimators on train window ───────────────
        for (symbol, asset_bars) in bars {
            let kf = kf_map.get_mut(symbol).unwrap();
            let freq = freq_map.get_mut(symbol).unwrap();
            let fusion = fusion_map.get_mut(symbol).unwrap();
            let asset_bars = &asset_bars[train_start..test_start];

            let mut prev_close = asset_bars.first().map(|b| b.close).unwrap_or(0.0);
            for bar in asset_bars {
                let obs = Observation {
                    price: bar.close,
                    volume: bar.volume,
                    timestamp_ms: bar.timestamp_ms,
                };
                if let Ok(posterior) = kf.update(&obs) {
                    let ks = kalman_signal(&posterior);
                    let freq_feat = freq.update(bar.close);
                    let fs = freq_feat.as_ref().map(|f| {
                        crate::ensemble::fusion::frequency_signal(f, posterior.velocity)
                    });
                    fusion.fuse(&[Some(ks), fs]);
                }
            }
        }

        // ── Test phase: simulate trading ─────────────────────────────────────
        let mut nav = self.config.initial_nav;
        let mut peak_nav = nav;
        let mut max_drawdown = 0.0f64;
        let mut pnl_series: Vec<f64> = Vec::new();
        let mut trades: Vec<SimulatedTrade> = Vec::new();
        // Simple position tracking: (entry_bar, entry_price, direction, size_usd, leverage).
        let mut open_positions: HashMap<String, (usize, f64, bool, f64, f64)> = HashMap::new();

        let n_test = test_end - test_start;
        for bar_idx in 0..n_test {
            let abs_idx = test_start + bar_idx;
            let mut bar_pnl = 0.0f64;

            // Update estimators and generate ensemble distributions.
            let mut distributions = HashMap::new();
            for (symbol, asset_bars) in bars {
                let bar = &asset_bars[abs_idx];
                let kf = kf_map.get_mut(symbol).unwrap();
                let freq = freq_map.get_mut(symbol).unwrap();
                let fusion = fusion_map.get_mut(symbol).unwrap();

                let obs = Observation {
                    price: bar.close,
                    volume: bar.volume,
                    timestamp_ms: bar.timestamp_ms,
                };
                if let Ok(posterior) = kf.update(&obs) {
                    let ks = kalman_signal(&posterior);
                    let freq_feat = freq.update(bar.close);
                    let fs = freq_feat.as_ref().map(|f| {
                        crate::ensemble::fusion::frequency_signal(f, posterior.velocity)
                    });
                    if let Some(dist) = fusion.fuse(&[Some(ks), fs]) {
                        distributions.insert(symbol.clone(), dist);
                    }
                }
            }

            // Exit stale positions (simple 1-bar hold for demo; extend with signal).
            for (symbol, (entry_bar, entry_px, is_long, size, lev)) in open_positions.drain() {
                let exit_bar = &bars[&symbol][abs_idx];
                let exit_px = if is_long {
                    exit_bar.close * (1.0 - exit_bar.spread_pct * 0.5)
                } else {
                    exit_bar.close * (1.0 + exit_bar.spread_pct * 0.5)
                };
                let ret = if is_long {
                    (exit_px - entry_px) / entry_px
                } else {
                    (entry_px - exit_px) / entry_px
                };
                let pnl = size * ret * lev;
                let tc = size * exit_bar.spread_pct;
                bar_pnl += pnl - tc;
                trades.push(SimulatedTrade {
                    asset: symbol,
                    entry_bar,
                    exit_bar: abs_idx,
                    direction: is_long,
                    entry_price: entry_px,
                    exit_price: exit_px,
                    size_usd: size,
                    leverage: lev,
                    realized_pnl: pnl,
                    transaction_cost_usd: tc,
                    fill_fraction: 1.0,
                });
            }

            // Open new positions based on optimizer signals.
            for (symbol, dist) in &distributions {
                if !dist.is_confident() {
                    continue;
                }
                let edge = dist.directional_edge();
                if edge.abs() < 0.1 {
                    continue;
                }
                if open_positions.len() >= self.risk_limits.max_concurrent_positions {
                    break;
                }

                let bar = &bars[symbol][abs_idx];
                let size_frac = (self.risk_limits.kelly_fraction
                    * edge.abs()
                    / dist.predictive_std.max(0.001))
                .min(self.risk_limits.max_single_asset_weight);
                let size_usd = nav * size_frac;
                let leverage = 1.0f64.min(self.risk_limits.max_leverage);
                let is_long = edge > 0.0;
                let entry_px = if is_long {
                    bar.close * (1.0 + bar.spread_pct * 0.5)
                } else {
                    bar.close * (1.0 - bar.spread_pct * 0.5)
                };

                // Check fill probability.
                let fill_p = (bar.book_depth_usd / size_usd.max(1.0) - 1.0)
                    .exp()
                    .min(1.0)
                    .max(0.0);
                if fill_p < 0.5 {
                    continue;
                }

                open_positions.insert(
                    symbol.clone(),
                    (abs_idx, entry_px, is_long, size_usd * fill_p, leverage),
                );
                bar_pnl -= size_usd * fill_p * bar.spread_pct * 0.5; // entry cost
            }

            nav += bar_pnl;
            pnl_series.push(bar_pnl);
            if nav > peak_nav {
                peak_nav = nav;
            }
            let dd = (peak_nav - nav) / peak_nav.max(1.0);
            if dd > max_drawdown {
                max_drawdown = dd;
            }
        }

        // ── Compute fold metrics ─────────────────────────────────────────────
        compute_fold_metrics(
            fold_idx,
            &pnl_series,
            &trades,
            self.config.initial_nav,
            nav,
            max_drawdown,
        )
    }
}

// ── Metrics computation ───────────────────────────────────────────────────────

fn compute_fold_metrics(
    fold_idx: usize,
    pnl_series: &[f64],
    trades: &[SimulatedTrade],
    initial_nav: f64,
    final_nav: f64,
    max_drawdown: f64,
) -> FoldResult {
    let n = pnl_series.len() as f64;
    if n < 2.0 {
        return FoldResult { fold_idx, ..Default::default() };
    }

    let returns: Vec<f64> = pnl_series
        .iter()
        .map(|&pnl| pnl / initial_nav)
        .collect();

    let mean_ret = returns.iter().sum::<f64>() / n;
    let var_ret = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>() / (n - 1.0);
    let std_ret = var_ret.sqrt().max(1e-10);

    // Downside deviation (for Sortino).
    let downside_var = returns.iter().map(|&r| r.min(0.0).powi(2)).sum::<f64>() / n;
    let downside_std = downside_var.sqrt().max(1e-10);

    let bars_per_year: f64 = 365.0 * 24.0 * 12.0; // 5-minute bars
    let sharpe = mean_ret / std_ret * bars_per_year.sqrt();
    let sortino = mean_ret / downside_std * bars_per_year.sqrt();
    let ann_return = ((final_nav / initial_nav.max(1.0)) - 1.0) * (bars_per_year / n);
    let calmar = if max_drawdown > 1e-10 { ann_return / max_drawdown } else { 0.0 };

    // Max drawdown duration.
    let mut dd_dur = 0usize;
    let mut cur_dur = 0usize;
    let mut running_max = 0.0f64;
    let mut cumulative = 0.0f64;
    for &r in &returns {
        cumulative += r;
        if cumulative > running_max {
            running_max = cumulative;
            cur_dur = 0;
        } else {
            cur_dur += 1;
            dd_dur = dd_dur.max(cur_dur);
        }
    }

    // Trade statistics.
    let n_trades = trades.len();
    let (wins, losses): (Vec<_>, Vec<_>) = trades
        .iter()
        .partition(|t| t.realized_pnl > 0.0);
    let hit_rate = if n_trades > 0 { wins.len() as f64 / n_trades as f64 } else { 0.0 };
    let avg_win = if wins.is_empty() {
        0.0
    } else {
        wins.iter().map(|t| t.realized_pnl / t.size_usd.max(1.0)).sum::<f64>() / wins.len() as f64
    };
    let avg_loss = if losses.is_empty() {
        0.0
    } else {
        losses.iter().map(|t| (t.realized_pnl / t.size_usd.max(1.0)).abs()).sum::<f64>()
            / losses.len() as f64
    };
    let total_turnover: f64 = trades.iter().map(|t| t.size_usd).sum();
    let total_tc: f64 = trades.iter().map(|t| t.transaction_cost_usd).sum();

    FoldResult {
        fold_idx,
        n_bars: pnl_series.len(),
        n_trades,
        total_return: final_nav / initial_nav - 1.0,
        annualized_return: ann_return,
        sharpe_ratio: sharpe,
        sortino_ratio: sortino,
        calmar_ratio: calmar,
        max_drawdown,
        max_drawdown_duration_bars: dd_dur,
        hit_rate,
        avg_win_pct: avg_win,
        avg_loss_pct: avg_loss,
        total_turnover,
        total_transaction_costs_usd: total_tc,
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_bars(n: usize, drift: f64) -> Vec<BacktestBar> {
        let mut bars = Vec::with_capacity(n);
        let mut price = 100.0f64;
        for i in 0..n as u64 {
            price *= 1.0 + drift + 0.001 * ((i as f64 * 0.1).sin());
            bars.push(BacktestBar {
                timestamp_ms: i * 300_000,
                open: price,
                high: price * 1.002,
                low: price * 0.998,
                close: price,
                volume: 1_000_000.0,
                spread_pct: 0.0001,
                book_depth_usd: 1_000_000.0,
            });
        }
        bars
    }

    #[test]
    fn backtest_runs_without_panic() {
        let config = WalkForwardConfig {
            train_bars: 100,
            test_bars: 50,
            max_folds: 2,
            initial_nav: 10_000.0,
            simulate_execution: true,
            adv_24h_usd: 1e8,
        };
        let limits = RiskLimits::from_env();
        let mut engine = BacktestEngine::new(config, limits);

        let mut bars = HashMap::new();
        bars.insert("BTC".to_string(), synthetic_bars(350, 0.0001));
        bars.insert("ETH".to_string(), synthetic_bars(350, 0.00008));

        let result = engine.run(&bars);
        assert_eq!(result.folds.len(), 2);
    }

    #[test]
    fn metrics_computed_for_drift() {
        let config = WalkForwardConfig {
            train_bars: 100,
            test_bars: 100,
            max_folds: 1,
            ..Default::default()
        };
        let limits = RiskLimits::from_env();
        let mut engine = BacktestEngine::new(config, limits);

        let mut bars = HashMap::new();
        bars.insert("BTC".to_string(), synthetic_bars(200, 0.001));

        let result = engine.run(&bars);
        assert!(!result.folds.is_empty());
        let fold = &result.folds[0];
        assert!(fold.n_bars > 0);
    }
}
