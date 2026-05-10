#![allow(unused_imports, unused_variables, dead_code)]

mod assets;
mod backtest;
mod config;
mod data;
mod ensemble;
mod error;
mod estimation;
mod execution;
mod features;
mod hmm;
mod ml;
mod optimizer;
mod orderbook;
mod risk;
mod signals;
mod stats;

use std::{collections::HashMap, sync::Arc, time::Duration};

use anyhow::Result;
use parking_lot::RwLock;
use tracing::{error, info, warn};

use assets::universe;
use config::Config;
use data::{
    ingestion::{bootstrap_historical, run_live},
    store::DataStore,
};
use ensemble::{
    EnsembleDistribution,
    fusion::{BayesianFusion, kalman_signal, frequency_signal, orderbook_signal},
};
use estimation::{
    Observation,
    StateEstimator,
    dual_kalman::DualKalmanFilter,
};
use execution::{
    executor::Executor,
    model::{ExecutionModel, ImpactParams},
};
use features::{
    assembler::{FeatureStore, assemble_row},
    frequency::FrequencyExtractor,
    kalman::KalmanFilter,
};
use optimizer::{AssetDistribution, OrderLeg};
use orderbook::hawkes::HawkesProcess;
use risk::{
    limits::{KillSwitchEvaluator, RiskLimits, EwmaVolEstimator, KillAction},
    manager::{PortfolioState, RiskManager},
};
use signals::generator::{Direction, TradeSignal, generate_combo};
use stats::{
    cointegration::{OuProcess, find_cointegrated_pairs, fit_ou},
    distribution::ReturnDist,
    garch::Garch11,
};

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let config = Config::from_env()?;
    info!("Starting quant_trader (testnet={})", config.use_testnet);

    let wallet: hypersdk::Address = config
        .wallet_address
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid wallet address: {}", e))?;

    // ── Shared state ──────────────────────────────────────────────────────────
    let store = Arc::new(RwLock::new(DataStore::new(2000)));
    let mut portfolio = PortfolioState::new(10_000.0);
    let executor = Executor::new(&config)?;

    // ── Risk + execution configuration ───────────────────────────────────────
    let risk_limits = RiskLimits::from_env();
    let exec_model = ExecutionModel::new(
        config.max_signal_age_ms,
        5.0,
        ImpactParams { eta: config.impact_eta, kappa: config.impact_kappa, adv_fraction: 0.001 },
    );

    // ── Historical bootstrap ──────────────────────────────────────────────────
    info!("Bootstrapping {} days of historical data…", config.history_days);
    bootstrap_historical(Arc::clone(&store), &config).await?;
    info!("Historical bootstrap complete");

    // ── Per-asset estimator state ─────────────────────────────────────────────
    let assets = universe();
    let n_sources = 3; // dual_kalman + frequency + orderbook

    let mut dual_kalman_filters: HashMap<String, DualKalmanFilter> = assets
        .iter()
        .map(|a| (a.symbol.clone(), DualKalmanFilter::new(config.dkf_sigma_obs, config.dkf_sigma_param_walk)))
        .collect();

    let mut freq_extractors: HashMap<String, FrequencyExtractor> = assets
        .iter()
        .map(|a| (a.symbol.clone(), FrequencyExtractor::new(config.freq_window_bars.max(64))))
        .collect();

    let mut bayesian_fusions: HashMap<String, BayesianFusion> = assets
        .iter()
        .map(|a| {
            let mut f = BayesianFusion::new(n_sources, config.fusion_evidence_decay, config.fusion_min_weight);
            f.register_source("dual_kalman");
            f.register_source("frequency");
            f.register_source("orderbook");
            (a.symbol.clone(), f)
        })
        .collect();

    // Legacy estimators (kept for feature assembly and cointegration).
    let mut kalman_filters: HashMap<String, KalmanFilter> = assets
        .iter()
        .map(|a| (a.symbol.clone(), KalmanFilter::default_crypto()))
        .collect();

    let mut garch_models: HashMap<String, Garch11> = HashMap::new();
    let mut dist_models: HashMap<String, ReturnDist> = HashMap::new();
    let mut ou_processes: Vec<OuProcess> = Vec::new();
    let mut hawkes_models: HashMap<String, HawkesProcess> = HashMap::new();
    let mut feature_store = FeatureStore::new(500);
    let mut ewma_vols: HashMap<String, EwmaVolEstimator> = assets
        .iter()
        .map(|a| (a.symbol.clone(), EwmaVolEstimator::new(0.94)))
        .collect();

    retrain_distributions(&store, &assets, &mut garch_models, &mut dist_models, &mut ou_processes);

    if let Err(e) = executor.refresh_portfolio(&mut portfolio, &wallet).await {
        warn!("Could not fetch initial portfolio state: {}; using default NAV", e);
    }

    // ── Live data task ────────────────────────────────────────────────────────
    let live_store = Arc::clone(&store);
    let live_config = config.clone();
    tokio::spawn(async move {
        loop {
            if let Err(e) = run_live(Arc::clone(&live_store), &live_config).await {
                error!("Live data task crashed: {}; restarting in 5s", e);
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        }
    });

    // ── Main trading loop ─────────────────────────────────────────────────────
    let interval = Duration::from_secs(config.signal_interval_secs);
    let mut tick_count = 0u64;
    let kill_switch_eval = KillSwitchEvaluator::new(&risk_limits);
    let risk_mgr = RiskManager::new(&risk_limits);

    loop {
        tokio::time::sleep(interval).await;
        tick_count += 1;

        info!("── Tick {} ─────────────────────────────────────", tick_count);

        // Weekly refit of GARCH / distributions.
        if tick_count % 2016 == 0 {
            info!("Weekly model refit…");
            retrain_distributions(&store, &assets, &mut garch_models, &mut dist_models, &mut ou_processes);
        }

        if let Err(e) = executor.refresh_portfolio(&mut portfolio, &wallet).await {
            warn!("Portfolio refresh failed: {}", e);
        }

        // ── Kill-switch evaluation ────────────────────────────────────────────
        let min_log_pred: f64 = dual_kalman_filters
            .values()
            .map(|kf| kf.posterior().log_predictive)
            .fold(0.0f64, f64::min);

        let worst_vol: f64 = ewma_vols.values().map(|e| e.current_vol()).fold(0.0f64, f64::max);

        if let Some(action) = kill_switch_eval.evaluate(
            portfolio.drawdown(),
            portfolio.leverage_ratio(),
            worst_vol,
            min_log_pred,
            portfolio.n_open_positions(),
        ) {
            match &action {
                KillAction::CloseAll { reason } => {
                    error!("KILL SWITCH CloseAll: {}", reason);
                    close_all_positions(&executor, &mut portfolio, &store).await;
                    continue;
                }
                KillAction::ReduceAll { fraction, reason } => {
                    warn!("KILL SWITCH ReduceAll({}): {}", fraction, reason);
                }
                KillAction::HaltNewTrades { reason } => {
                    warn!("KILL SWITCH HaltNewTrades: {}", reason);
                    monitor_positions(&executor, &mut portfolio, &store, &risk_limits).await;
                    log_portfolio_state(&portfolio);
                    continue;
                }
            }
        }

        // ── Per-asset: update estimators, build AssetDistributions ───────────
        let btc_returns: Vec<f64> = store.read().returns("BTC", "5m");
        let mut asset_dists: Vec<AssetDistribution> = Vec::new();
        let mut latest_rows: HashMap<String, features::assembler::FeatureRow> = HashMap::new();

        for asset in &assets {
            let sym = &asset.symbol;

            let price = {
                let s = store.read();
                s.bars_for(sym, "5m")
                    .and_then(|b| b.back())
                    .map(|b| b.close)
                    .unwrap_or(0.0)
            };
            if price <= 0.0 { continue; }

            // EWMA vol update.
            let log_ret = {
                let s = store.read();
                s.returns(sym, "5m").last().copied().unwrap_or(0.0)
            };
            if let Some(ev) = ewma_vols.get_mut(sym) {
                ev.update(log_ret);
            }

            // ── Dual Kalman Filter ────────────────────────────────────────────
            let volume = store.read()
                .bars_for(sym, "5m")
                .and_then(|b| b.back())
                .map(|b| b.volume)
                .unwrap_or(0.0);
            let now_ms = timestamp_now_ms();
            let obs = Observation { price, volume, timestamp_ms: now_ms };
            let dkf = dual_kalman_filters.get_mut(sym).unwrap();
            let posterior = match dkf.update(&obs) {
                Ok(p) => p,
                Err(e) => { warn!("DKF update failed for {}: {}", sym, e); continue; }
            };

            // ── Frequency extractor ───────────────────────────────────────────
            let freq_ext = freq_extractors.get_mut(sym).unwrap();
            let freq_feat = freq_ext.update(price);

            // ── Hawkes / order book signal ────────────────────────────────────
            let hawkes_opt = estimate_hawkes(&store, sym, &mut hawkes_models);

            // ── Bayesian fusion ───────────────────────────────────────────────
            let ks = kalman_signal(&posterior);
            let fs = freq_feat.as_ref().map(|f| frequency_signal(f, posterior.velocity));
            let obs_signal = hawkes_opt.as_ref().map(|h| orderbook_signal(h));

            let fusion = bayesian_fusions.get_mut(sym).unwrap();
            if let Some(dist) = fusion.fuse(&[Some(ks), fs, obs_signal]) {
                // TC estimate from live order book spread + fixed impact buffer.
                let spread_bps = store.read()
                    .books
                    .get(sym)
                    .and_then(|b| b.spread_bps())
                    .unwrap_or(10.0);
                let estimated_tc_bps = spread_bps + 2.0;

                // CVaR from fitted return distribution.
                let cvar_95 = dist_models.get(sym).map(|d| d.cvar(0.95).abs()).unwrap_or(0.0);
                let cvar_99 = dist_models.get(sym).map(|d| d.cvar(0.99).abs()).unwrap_or(0.0);

                // Signed current position fraction.
                let current_fraction = portfolio.positions.get(sym)
                    .map(|p| {
                        let sign = if p.direction.is_long() { 1.0 } else { -1.0 };
                        sign * p.size_usd / portfolio.nav.max(1.0)
                    })
                    .unwrap_or(0.0);

                asset_dists.push(AssetDistribution {
                    asset: sym.clone(),
                    ensemble: dist,
                    current_fraction,
                    entry_price: posterior.level,
                    estimated_tc_bps,
                    cvar_95,
                    cvar_99,
                });
            }

            // ── Legacy feature row (for feature store) ────────────────────────
            let garch_vol = garch_models
                .get(sym)
                .map(|g| {
                    let rets = store.read().returns(sym, "5m");
                    g.forecast_sigma(&rets)
                })
                .unwrap_or(0.01);

            let ou_z = ou_processes
                .iter()
                .find(|ou| ou.asset1 == *sym || ou.asset2 == *sym)
                .map(|ou| {
                    let s = store.read();
                    let p1 = s.bars_for(&ou.asset1, "5m").and_then(|b| b.back()).map(|b| b.close).unwrap_or(0.0);
                    let p2 = s.bars_for(&ou.asset2, "5m").and_then(|b| b.back()).map(|b| b.close).unwrap_or(0.0);
                    ou.z_score(p1, p2)
                })
                .unwrap_or(0.0);

            let hawkes_ratio = hawkes_models.get(sym).map(|h| h.cross_excitation_ratio()).unwrap_or(0.0);
            let kf = kalman_filters.get_mut(sym).unwrap();

            let row = {
                let s = store.read();
                assemble_row(asset, &s, kf, garch_vol, ou_z, &btc_returns, hawkes_ratio, freq_feat)
            };
            if let Some(row) = row {
                latest_rows.insert(sym.clone(), row.clone());
                feature_store.push(row);
            }
        }

        // ── Joint combo optimization ──────────────────────────────────────────
        let combo = match generate_combo(&asset_dists, &portfolio, &config) {
            Some(c) => c,
            None => {
                info!("No combo order generated this tick");
                monitor_positions(&executor, &mut portfolio, &store, &risk_limits).await;
                log_portfolio_state(&portfolio);
                continue;
            }
        };

        info!(
            "Combo order: {} leg(s), ELG={:.5}, TC={:.2}bps, regime={}",
            combo.n_legs(),
            combo.total_expected_log_growth,
            combo.total_tc_bps,
            combo.regime_summary()
        );

        // ── Risk checks and execution per leg ─────────────────────────────────
        for leg in &combo.legs {
            let asset_idx = assets.iter().find(|a| a.symbol == leg.asset).map(|a| a.index).unwrap_or(0);
            let signal = leg_to_signal(leg, asset_idx);

            let dist = dist_models.get(&leg.asset).map(|d| d as &ReturnDist);
            match risk_mgr.check_signal(&signal, &portfolio, dist, None) {
                Ok(()) => {
                    let adv_24h = store.read()
                        .bars_for(&leg.asset, "1h")
                        .map(|b| b.iter().rev().take(24).map(|bar| bar.volume * bar.close).sum::<f64>())
                        .unwrap_or(1e8);
                    let spread_pct = store.read()
                        .books
                        .get(&leg.asset)
                        .and_then(|b| b.spread_bps())
                        .map(|s| s / 10_000.0)
                        .unwrap_or(0.0001);
                    let depth_usd = store.read()
                        .books
                        .get(&leg.asset)
                        .map(|b| b.bid_depth(10) + b.ask_depth(10))
                        .unwrap_or(1e6);

                    let exec_metrics = exec_model.evaluate(&signal, timestamp_now_ms(), adv_24h, spread_pct, depth_usd);
                    if exec_metrics.is_executable {
                        info!(
                            "Executing leg: {} {} (fill_p={:.1}%, impact={:.2}bps)",
                            leg.direction, leg.asset,
                            exec_metrics.fill_probability * 100.0,
                            exec_metrics.total_cost_pct * 10_000.0,
                        );
                        execute_signal(&executor, &signal, &mut portfolio, &store).await;
                    } else {
                        warn!("Leg execution rejected for {}: {}", leg.asset, exec_metrics.rejection_reason);
                    }
                }
                Err(e) => warn!("Risk check failed for {} leg: {}", leg.asset, e),
            }
        }

        monitor_positions(&executor, &mut portfolio, &store, &risk_limits).await;
        log_portfolio_state(&portfolio);
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn timestamp_now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Convert an `OrderLeg` to a `TradeSignal` for risk checking and execution.
///
/// The `TradeSignal` interface is used by `RiskManager` and `Executor` which
/// pre-date the combo architecture. Ensemble fields are approximated from the
/// leg's direction and confidence since the full distribution is in the combo.
fn leg_to_signal(leg: &OrderLeg, asset_index: usize) -> TradeSignal {
    let now_ms = timestamp_now_ms();
    let (p_up, p_down) = match leg.direction {
        Direction::Long => (leg.confidence, 1.0 - leg.confidence),
        Direction::Short => (1.0 - leg.confidence, leg.confidence),
    };
    TradeSignal {
        signal_id: uuid::Uuid::new_v4().to_string(),
        generated_at_ms: now_ms,
        asset: leg.asset.clone(),
        asset_index,
        direction: leg.direction,
        entry_price: leg.entry_price,
        stop_loss: leg.stop_loss,
        take_profit: leg.take_profit,
        leverage: leg.leverage,
        position_size_usd: leg.position_size_usd,
        expected_value: leg.expected_log_growth,
        signal_confidence: leg.confidence,
        directional_edge: if leg.is_long() { leg.confidence } else { -leg.confidence },
        predicted_vol_4h: 0.0,
        regime_label: leg.regime_label.clone(),
        ensemble_p_up: p_up,
        ensemble_p_down: p_down,
        ensemble_confidence: leg.confidence,
    }
}

async fn execute_signal(
    executor: &Executor,
    signal: &TradeSignal,
    portfolio: &mut PortfolioState,
    store: &Arc<RwLock<DataStore>>,
) {
    let oid = match executor.place_entry(signal).await {
        Ok(id) => { info!("Entry OID={} for {} {}", id, signal.direction, signal.asset); id }
        Err(e) => { error!("Failed to place entry for {}: {}", signal.asset, e); return; }
    };

    let sl_oid = executor.place_stop_loss(signal).await.ok();
    let tp_oid = executor.place_take_profit(signal).await.ok();

    let pos = risk::manager::Position {
        asset: signal.asset.clone(),
        direction: signal.direction,
        entry_price: signal.entry_price,
        size_usd: signal.position_size_usd,
        leverage: signal.leverage,
        stop_loss: signal.stop_loss,
        take_profit: signal.take_profit,
        open_time_ms: timestamp_now_ms(),
        oid: Some(oid),
        sl_oid,
        tp_oid,
    };
    portfolio.positions.insert(signal.asset.clone(), pos);
}

async fn monitor_positions(
    executor: &Executor,
    portfolio: &mut PortfolioState,
    store: &Arc<RwLock<DataStore>>,
    limits: &RiskLimits,
) {
    let now_ms = timestamp_now_ms();
    let max_hold_ms = 48 * 3_600_000u64;
    let risk = RiskManager::new(limits);

    let to_close: Vec<String> = portfolio
        .positions
        .iter()
        .filter_map(|(asset, pos)| {
            let current_px = {
                let s = store.read();
                s.books.get(asset).and_then(|b| b.mid_price()).unwrap_or(pos.entry_price)
            };
            risk.should_close(pos, now_ms, current_px, max_hold_ms).map(|reason| {
                info!("Closing {} position: {}", asset, reason);
                asset.clone()
            })
        })
        .collect();

    for asset in to_close {
        if let Some(pos) = portfolio.positions.remove(&asset) {
            if let Some(sl_oid) = pos.sl_oid {
                let _ = executor.cancel_order(pos_asset_index(&asset), sl_oid).await;
            }
            if let Some(tp_oid) = pos.tp_oid {
                let _ = executor.cancel_order(pos_asset_index(&asset), tp_oid).await;
            }
            if let Err(e) = executor.close_position(&pos, store).await {
                error!("Failed to close {}: {}", asset, e);
                portfolio.positions.insert(asset, pos);
            }
        }
    }
}

async fn close_all_positions(
    executor: &Executor,
    portfolio: &mut PortfolioState,
    store: &Arc<RwLock<DataStore>>,
) {
    let assets: Vec<String> = portfolio.positions.keys().cloned().collect();
    for asset in assets {
        if let Some(pos) = portfolio.positions.remove(&asset) {
            if let Some(sl_oid) = pos.sl_oid {
                let _ = executor.cancel_order(pos_asset_index(&asset), sl_oid).await;
            }
            if let Some(tp_oid) = pos.tp_oid {
                let _ = executor.cancel_order(pos_asset_index(&asset), tp_oid).await;
            }
            if let Err(e) = executor.close_position(&pos, store).await {
                error!("Emergency close failed for {}: {}", asset, e);
                portfolio.positions.insert(asset, pos);
            }
        }
    }
}

fn pos_asset_index(asset: &str) -> usize {
    universe().iter().find(|a| a.symbol == asset).map(|a| a.index).unwrap_or(0)
}

fn estimate_hawkes(
    store: &Arc<RwLock<DataStore>>,
    asset: &str,
    hawkes_models: &mut HashMap<String, HawkesProcess>,
) -> Option<HawkesProcess> {
    let (buy_times, sell_times) = {
        let s = store.read();
        let hist = s.trades.get(asset)?;
        let buy: Vec<u64> = hist.buy_times_ms.iter().copied().collect();
        let sell: Vec<u64> = hist.sell_times_ms.iter().copied().collect();
        (buy, sell)
    };

    if buy_times.len() < 10 || sell_times.len() < 10 {
        return hawkes_models.get(asset).cloned();
    }

    match HawkesProcess::fit(&buy_times, &sell_times) {
        Ok(h) => {
            let cloned = h.clone();
            hawkes_models.insert(asset.to_string(), h);
            Some(cloned)
        }
        Err(_) => hawkes_models.get(asset).cloned(),
    }
}

fn retrain_distributions(
    store: &Arc<RwLock<DataStore>>,
    assets: &[assets::Asset],
    garch_models: &mut HashMap<String, Garch11>,
    dist_models: &mut HashMap<String, ReturnDist>,
    ou_processes: &mut Vec<OuProcess>,
) {
    let s = store.read();

    for asset in assets {
        let returns = s.returns(&asset.symbol, "1h");
        if returns.len() >= 50 {
            match Garch11::fit(&returns) {
                Ok(g) => {
                    info!("GARCH {}: ω={:.2e} α={:.3} β={:.3}", asset.symbol, g.omega, g.alpha, g.beta);
                    garch_models.insert(asset.symbol.clone(), g);
                }
                Err(e) => warn!("GARCH fit failed for {}: {}", asset.symbol, e),
            }
        }
        if returns.len() >= 30 {
            match ReturnDist::fit_best(&returns) {
                Ok(d) => {
                    info!("ReturnDist {}: {} (AIC={:.1})", asset.symbol, d.name(), d.aic());
                    dist_models.insert(asset.symbol.clone(), d);
                }
                Err(e) => warn!("ReturnDist fit failed for {}: {}", asset.symbol, e),
            }
        }
    }

    let price_series: Vec<(String, Vec<f64>)> = assets
        .iter()
        .filter_map(|a| {
            let closes = s.closes(&a.symbol, "1d");
            if closes.len() >= 30 { Some((a.symbol.clone(), closes)) } else { None }
        })
        .collect();

    let coint = stats::cointegration::find_cointegrated_pairs(&price_series);
    ou_processes.clear();
    for cr in &coint {
        if cr.is_cointegrated {
            let p1 = s.closes(&cr.asset1, "1d");
            let p2 = s.closes(&cr.asset2, "1d");
            if let Ok(ou) = fit_ou(cr, &p1, &p2) {
                info!("OU {}/{}: κ={:.3} half_life={:.1}d", ou.asset1, ou.asset2, ou.kappa, ou.half_life);
                ou_processes.push(ou);
            }
        }
    }
}

fn log_portfolio_state(portfolio: &PortfolioState) {
    info!(
        "Portfolio: NAV={:.0} | Leverage={:.2}x | Drawdown={:.1}% | Positions={}",
        portfolio.nav,
        portfolio.leverage_ratio(),
        portfolio.drawdown() * 100.0,
        portfolio.n_open_positions(),
    );
}
