#![allow(unused_imports, unused_variables, dead_code)]
mod assets;
mod config;
mod data;
mod error;
mod execution;
mod features;
mod hmm;
mod ml;
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
use execution::Executor;
use features::{
    assembler::{FeatureStore, assemble_row},
    kalman::KalmanFilter,
};
use hmm::model::{HmmModel, RegimeState};
use ml::ensemble::{EnsembleOutput, ModelEnsemble};
use orderbook::hawkes::HawkesProcess;
use risk::manager::{PortfolioState, RiskManager};
use signals::generator::generate_signals;
use stats::{
    cointegration::{OuProcess, find_cointegrated_pairs, fit_ou},
    distribution::ReturnDist,
    garch::Garch11,
};

/// HMM observation dimension (see hmm/model.rs).
const HMM_OBS_DIM: usize = 7;

#[tokio::main]
async fn main() -> Result<()> {
    // ── Setup ────────────────────────────────────────────────────────────────
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

    // ── Shared state ────────────────────────────────────────────────────────
    let store = Arc::new(RwLock::new(DataStore::new(2000)));
    let mut portfolio = PortfolioState::new(10_000.0); // Bootstrapped; refreshed from API.
    let executor = Executor::new(&config)?;

    // ── Historical data bootstrap ────────────────────────────────────────────
    info!("Bootstrapping {} days of historical data…", config.history_days);
    bootstrap_historical(Arc::clone(&store), &config).await?;
    info!("Historical bootstrap complete");

    // ── Initial model training ───────────────────────────────────────────────
    let assets = universe();
    let mut kalman_filters: HashMap<String, KalmanFilter> = assets
        .iter()
        .map(|a| (a.symbol.clone(), KalmanFilter::default_crypto()))
        .collect();

    let mut garch_models: HashMap<String, Garch11> = HashMap::new();
    let mut dist_models: HashMap<String, ReturnDist> = HashMap::new();
    let mut ou_processes: Vec<OuProcess> = Vec::new();
    let mut hawkes_models: HashMap<String, HawkesProcess> = HashMap::new();
    let mut feature_store = FeatureStore::new(500);
    let mut ml_ensemble = ModelEnsemble::new(
        config.gb_n_estimators,
        config.gb_learning_rate,
        config.gb_max_depth,
    );
    let mut hmm = HmmModel::new(HMM_OBS_DIM);
    let mut current_regime = RegimeState {
        regime: hmm::model::Regime::Consolidation,
        probability: 0.6,
        posterior: vec![0.1, 0.1, 0.7, 0.1],
    };

    // Initial training pass.
    retrain_all(
        &store,
        &assets,
        &mut kalman_filters,
        &mut garch_models,
        &mut dist_models,
        &mut ou_processes,
        &mut feature_store,
        &mut ml_ensemble,
        &mut hmm,
        &config,
    );

    // Refresh portfolio NAV from the exchange.
    if let Err(e) = executor.refresh_portfolio(&mut portfolio, &wallet).await {
        warn!("Could not fetch initial portfolio state: {}; using default NAV", e);
    }

    // ── Live data task (background) ──────────────────────────────────────────
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

    loop {
        tokio::time::sleep(interval).await;
        tick_count += 1;

        info!("── Tick {} ─────────────────────────────────────", tick_count);

        // Weekly retraining (every 2016 ticks at 5m = 7 days).
        if tick_count % 2016 == 0 {
            info!("Weekly model retraining…");
            retrain_all(
                &store,
                &assets,
                &mut kalman_filters,
                &mut garch_models,
                &mut dist_models,
                &mut ou_processes,
                &mut feature_store,
                &mut ml_ensemble,
                &mut hmm,
                &config,
            );
        }

        // Refresh portfolio.
        if let Err(e) = executor.refresh_portfolio(&mut portfolio, &wallet).await {
            warn!("Portfolio refresh failed: {}", e);
        }

        // ── Feature computation ──────────────────────────────────────────────
        let btc_returns: Vec<f64> = {
            let s = store.read();
            s.returns("BTC", "5m")
        };

        let mut ml_outputs: HashMap<String, EnsembleOutput> = HashMap::new();
        let mut latest_rows: HashMap<String, features::assembler::FeatureRow> = HashMap::new();

        for asset in &assets {
            let sym = &asset.symbol;

            // Hawkes estimation (if enough data).
            let hawkes_cross_ratio = estimate_hawkes(&store, sym, &mut hawkes_models);

            // GARCH conditional vol.
            let garch_vol = garch_models
                .get(sym)
                .map(|g| {
                    let rets = store.read().returns(sym, "5m");
                    if rets.is_empty() {
                        0.01
                    } else {
                        g.forecast_sigma(&rets)
                    }
                })
                .unwrap_or(0.01);

            // OU z-score for this asset's best spread.
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

            let kf = kalman_filters.get_mut(sym).unwrap();
            let row = {
                let s = store.read();
                assemble_row(asset, &s, kf, garch_vol, ou_z, &btc_returns, hawkes_cross_ratio)
            };

            if let Some(row) = row {
                latest_rows.insert(sym.clone(), row.clone());
                feature_store.push(row.clone());

                // ML inference.
                if ml_ensemble.is_trained(sym) {
                    if let Some(output) = ml_ensemble.predict(sym, &row) {
                        ml_outputs.insert(sym.clone(), output);
                    }
                }
            }
        }

        // ── HMM regime update ────────────────────────────────────────────────
        let btc_out = ml_outputs.get("BTC");
        let eth_out = ml_outputs.get("ETH");
        if let (Some(btc), Some(eth)) = (btc_out, eth_out) {
            let obs = hmm_observation(btc, eth, &ml_outputs, &portfolio);
            // Build a short rolling sequence for the forward pass (last 20 obs).
            let obs_seq = vec![obs]; // Online: single-step update.
            current_regime = hmm.filter_last(&obs_seq);
            info!(
                "Regime: {} (p={:.2})",
                current_regime.regime.label(),
                current_regime.probability
            );
        }

        // ── Signal generation ────────────────────────────────────────────────
        let signals = generate_signals(
            &assets,
            &ml_outputs,
            &latest_rows,
            &current_regime,
            &portfolio,
            &config,
            &dist_models,
        );

        // ── Risk checks and execution ────────────────────────────────────────
        let risk_mgr = RiskManager::new(&config);

        for signal in &signals {
            info!("Signal: {}", serde_json::to_string_pretty(&signal.to_json()).unwrap_or_default());

            let dist = dist_models.get(&signal.asset).map(|d| d as &ReturnDist);
            match risk_mgr.check_signal(signal, &portfolio, dist) {
                Ok(()) => {
                    info!("Risk check passed for {} {}", signal.direction, signal.asset);
                    execute_signal(&executor, signal, &mut portfolio, &store).await;
                }
                Err(e) => {
                    warn!("Risk check failed for {}: {}", signal.asset, e);
                }
            }
        }

        // ── Monitor open positions ───────────────────────────────────────────
        monitor_positions(&executor, &mut portfolio, &store).await;

        log_portfolio_state(&portfolio);
    }
}

// ─── Execution helper ─────────────────────────────────────────────────────────

async fn execute_signal(
    executor: &Executor,
    signal: &signals::generator::TradeSignal,
    portfolio: &mut PortfolioState,
    store: &Arc<RwLock<DataStore>>,
) {
    // Place entry order.
    let oid = match executor.place_entry(signal).await {
        Ok(id) => {
            info!("Entry order placed OID={} for {} {}", id, signal.direction, signal.asset);
            id
        }
        Err(e) => {
            error!("Failed to place entry for {}: {}", signal.asset, e);
            return;
        }
    };

    // Place stop-loss.
    let sl_oid = executor.place_stop_loss(signal).await.ok();
    let tp_oid = executor.place_take_profit(signal).await.ok();

    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);

    let pos = risk::manager::Position {
        asset: signal.asset.clone(),
        direction: signal.direction,
        entry_price: signal.entry_price,
        size_usd: signal.position_size_usd,
        leverage: signal.leverage,
        stop_loss: signal.stop_loss,
        take_profit: signal.take_profit,
        open_time_ms: now_ms,
        oid: Some(oid),
        sl_oid,
        tp_oid,
    };
    portfolio.positions.insert(signal.asset.clone(), pos);
}

// ─── Position monitoring ──────────────────────────────────────────────────────

async fn monitor_positions(
    executor: &Executor,
    portfolio: &mut PortfolioState,
    store: &Arc<RwLock<DataStore>>,
) {
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);

    let fallback_config = Config {
        private_key: String::new(),
        wallet_address: String::new(),
        use_testnet: false,
        max_portfolio_leverage: 3.0,
        max_single_asset_weight: 0.3,
        daily_drawdown_limit: 0.08,
        target_daily_vol: 0.015,
        kelly_fraction: 0.3,
        hmm_states: 4,
        gb_n_estimators: 100,
        gb_learning_rate: 0.1,
        gb_max_depth: 3,
        history_days: 180,
        signal_interval_secs: 300,
    };
    let monitor_config = Config::from_env().unwrap_or(fallback_config);
    let risk = RiskManager::new(&monitor_config);

    let to_close: Vec<String> = portfolio
        .positions
        .iter()
        .filter_map(|(asset, pos)| {
            let current_px = {
                let s = store.read();
                s.books
                    .get(asset)
                    .and_then(|b| b.mid_price())
                    .unwrap_or(pos.entry_price)
            };
            risk.should_close(pos, now_ms, current_px).map(|reason| {
                info!("Closing {} position: {}", asset, reason);
                asset.clone()
            })
        })
        .collect();

    for asset in to_close {
        if let Some(pos) = portfolio.positions.remove(&asset) {
            // Cancel outstanding SL and TP orders.
            if let Some(sl_oid) = pos.sl_oid {
                let _ = executor.cancel_order(pos_asset_index(&asset), sl_oid).await;
            }
            if let Some(tp_oid) = pos.tp_oid {
                let _ = executor.cancel_order(pos_asset_index(&asset), tp_oid).await;
            }
            // Close at market.
            if let Err(e) = executor.close_position(&pos, store).await {
                error!("Failed to close {}: {}", asset, e);
                portfolio.positions.insert(asset, pos);
            }
        }
    }
}

fn pos_asset_index(asset: &str) -> usize {
    universe()
        .iter()
        .find(|a| a.symbol == asset)
        .map(|a| a.index)
        .unwrap_or(0)
}

// ─── Full retraining ──────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn retrain_all(
    store: &Arc<RwLock<DataStore>>,
    assets: &[assets::Asset],
    kalman_filters: &mut HashMap<String, KalmanFilter>,
    garch_models: &mut HashMap<String, Garch11>,
    dist_models: &mut HashMap<String, ReturnDist>,
    ou_processes: &mut Vec<OuProcess>,
    feature_store: &mut FeatureStore,
    ml_ensemble: &mut ModelEnsemble,
    hmm: &mut HmmModel,
    config: &Config,
) {
    let s = store.read();

    // GARCH + return distribution fitting.
    for asset in assets {
        let returns = s.returns(&asset.symbol, "1h");
        if returns.len() >= 50 {
            match Garch11::fit(&returns) {
                Ok(g) => {
                    info!(
                        "GARCH {} fitted: ω={:.2e} α={:.3} β={:.3} ν={:.1}",
                        asset.symbol, g.omega, g.alpha, g.beta, g.nu
                    );
                    garch_models.insert(asset.symbol.clone(), g);
                }
                Err(e) => warn!("GARCH fit failed for {}: {}", asset.symbol, e),
            }
        }

        // Fit the best return distribution (Student-t / Cauchy / Discrete).
        if returns.len() >= 30 {
            match ReturnDist::fit_best(&returns) {
                Ok(d) => {
                    info!(
                        "ReturnDist {} → {} (AIC={:.1} KS={:.4})",
                        asset.symbol,
                        d.name(),
                        d.aic(),
                        d.ks_stat(),
                    );
                    dist_models.insert(asset.symbol.clone(), d);
                }
                Err(e) => warn!("ReturnDist fit failed for {}: {}", asset.symbol, e),
            }
        }
    }

    // Cointegration.
    let price_series: Vec<(String, Vec<f64>)> = assets
        .iter()
        .filter_map(|a| {
            let closes = s.closes(&a.symbol, "1d");
            if closes.len() >= 30 {
                Some((a.symbol.clone(), closes))
            } else {
                None
            }
        })
        .collect();

    let coint_results = stats::cointegration::find_cointegrated_pairs(&price_series);
    ou_processes.clear();
    for cr in &coint_results {
        if cr.is_cointegrated {
            let p1 = s.closes(&cr.asset1, "1d");
            let p2 = s.closes(&cr.asset2, "1d");
            if let Ok(ou) = fit_ou(cr, &p1, &p2) {
                info!(
                    "OU {}/{}: κ={:.3} θ={:.3} half_life={:.1}d",
                    ou.asset1, ou.asset2, ou.kappa, ou.theta, ou.half_life
                );
                ou_processes.push(ou);
            }
        }
    }

    drop(s);

    // Rebuild feature store and train ML models.
    // Walk forward to populate feature_store with historical feature rows.
    info!("Rebuilding feature store from history…");
    for asset in assets {
        let kf = kalman_filters.get_mut(&asset.symbol).unwrap();
        let btc_returns: Vec<f64> = store.read().returns("BTC", "5m");

        let n_bars = store.read().bars_for(&asset.symbol, "5m").map(|b| b.len()).unwrap_or(0);
        // We can only assemble a row for the current state; walk-forward
        // would require re-running Kalman on each prefix. Here we assemble
        // the latest row and rely on historical training data collected during
        // live operation over time.
        let garch_vol = garch_models
            .get(&asset.symbol)
            .map(|g| {
                let rets = store.read().returns(&asset.symbol, "5m");
                g.forecast_sigma(&rets)
            })
            .unwrap_or(0.01);

        let ou_z = ou_processes
            .iter()
            .find(|ou| ou.asset1 == asset.symbol || ou.asset2 == asset.symbol)
            .map(|ou| {
                let s = store.read();
                let p1 = s.bars_for(&ou.asset1, "5m").and_then(|b| b.back()).map(|b| b.close).unwrap_or(0.0);
                let p2 = s.bars_for(&ou.asset2, "5m").and_then(|b| b.back()).map(|b| b.close).unwrap_or(0.0);
                ou.z_score(p1, p2)
            })
            .unwrap_or(0.0);

        let row = {
            let s = store.read();
            assemble_row(asset, &s, kf, garch_vol, ou_z, &btc_returns, 0.0)
        };
        if let Some(row) = row {
            feature_store.push(row);
        }
    }

    // Train ML models if enough feature rows are available (horizon=48 bars ≈ 4h).
    for asset in assets {
        let rows = feature_store.rows.get(&asset.symbol);
        if let Some(rows) = rows {
            if rows.len() >= 68 {
                match ml_ensemble.train(&asset.symbol, rows, 48) {
                    Ok(()) => info!("ML ensemble trained for {}", asset.symbol),
                    Err(e) => warn!("ML training failed for {}: {}", asset.symbol, e),
                }
            } else {
                info!("{}: need {} more feature rows for ML training", asset.symbol, 68 - rows.len());
            }
        }
    }

    // HMM training — requires at least 20 observation steps.
    // We build synthetic observations from GARCH vols and GARCH means as a proxy
    // during bootstrapping; proper HMM training happens after live data accumulates.
    info!("HMM pre-training skipped until enough live observations are collected");
}

fn estimate_hawkes(
    store: &Arc<RwLock<DataStore>>,
    asset: &str,
    hawkes_models: &mut HashMap<String, HawkesProcess>,
) -> f64 {
    let (buy_times, sell_times) = {
        let s = store.read();
        if let Some(hist) = s.trades.get(asset) {
            let buy: Vec<u64> = hist.buy_times_ms.iter().copied().collect();
            let sell: Vec<u64> = hist.sell_times_ms.iter().copied().collect();
            (buy, sell)
        } else {
            return 0.0;
        }
    };

    if buy_times.len() < 10 || sell_times.len() < 10 {
        return hawkes_models.get(asset).map(|h| h.cross_excitation_ratio()).unwrap_or(0.0);
    }

    match HawkesProcess::fit(&buy_times, &sell_times) {
        Ok(h) => {
            let ratio = h.cross_excitation_ratio();
            hawkes_models.insert(asset.to_string(), h);
            ratio
        }
        Err(_) => hawkes_models.get(asset).map(|h| h.cross_excitation_ratio()).unwrap_or(0.0),
    }
}

fn hmm_observation(
    btc: &EnsembleOutput,
    eth: &EnsembleOutput,
    all: &HashMap<String, EnsembleOutput>,
    portfolio: &PortfolioState,
) -> Vec<f64> {
    let vol_agg = all.values().map(|o| o.predicted_vol_4h).sum::<f64>()
        / all.len().max(1) as f64;

    // Cross-asset correlation: average pairwise abs correlation (proxy).
    let corr_proxy = 0.7f64; // Would be computed from DataStore in production.

    let exposure = portfolio.leverage_ratio() / 5.0; // Normalised to [0,1] assuming max 5x.

    let funding_z_mean = 0.0f64; // Would be taken from DataStore funding history.

    let n_pos = portfolio.n_open_positions() as f64 / 5.0; // Normalised to [0,1].

    vec![
        btc.directional_edge,
        eth.directional_edge,
        vol_agg,
        corr_proxy,
        exposure,
        funding_z_mean,
        n_pos,
    ]
}

fn log_portfolio_state(portfolio: &PortfolioState) {
    info!(
        "Portfolio: NAV={:.0} | Leverage={:.2}× | Drawdown={:.1}% | Positions={}",
        portfolio.nav,
        portfolio.leverage_ratio(),
        portfolio.drawdown() * 100.0,
        portfolio.n_open_positions(),
    );
}
