use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;
use futures::StreamExt;
use hypersdk::hypercore::{
    self,
    types::{AssetContext, CandleInterval, Incoming, Subscription},
    NonceHandler,
};
use parking_lot::RwLock;
use tracing::{error, info, warn};

use crate::{
    assets::{universe, Asset},
    config::Config,
    data::store::{AssetContextSnapshot, Bar, DataStore, FundingSample},
};

const INTERVALS: &[(&str, CandleInterval)] = &[
    ("1m", CandleInterval::OneMinute),
    ("5m", CandleInterval::FiveMinutes),
    ("1h", CandleInterval::OneHour),
    ("1d", CandleInterval::OneDay),
];

/// Bootstrap: fetch historical OHLCV for all assets and intervals.
pub async fn bootstrap_historical(
    store: Arc<RwLock<DataStore>>,
    config: &Config,
) -> Result<()> {
    let client = if config.use_testnet {
        hypercore::testnet()
    } else {
        hypercore::mainnet()
    };

    let assets = universe();
    let end_ms = Utc::now().timestamp_millis() as u64;
    let start_ms = end_ms - config.history_days * 86_400_000;

    for asset in &assets {
        for &(label, interval) in INTERVALS {
            info!("Fetching {}:{} history ({} days)", asset.symbol, label, config.history_days);
            match client
                .candle_snapshot(&asset.symbol, interval, start_ms, end_ms)
                .await
            {
                Ok(candles) => {
                    let mut prev_close: Option<f64> = None;
                    let mut store = store.write();
                    for c in &candles {
                        let bar = Bar::from_candle(c, prev_close);
                        prev_close = Some(bar.close);
                        store.push_bar(&asset.symbol, label, bar);
                    }
                    info!(
                        "Loaded {} candles for {}:{}",
                        candles.len(),
                        asset.symbol,
                        label
                    );
                }
                Err(e) => {
                    error!("Failed to fetch {}:{}: {}", asset.symbol, label, e);
                }
            }
        }

        // Fetch funding history (last 7 days sampled at 8h intervals).
        let funding_start = end_ms - 7 * 86_400_000;
        if let Ok(rates) = client.funding_history(&asset.symbol, funding_start, None).await {
            let mut store = store.write();
            for r in &rates {
                use rust_decimal::prelude::ToPrimitive;
                let sample = FundingSample {
                    time: r.time,
                    rate: r.funding_rate.to_f64().unwrap_or(0.0),
                    premium: r.premium.to_f64().unwrap_or(0.0),
                };
                store.push_funding(&asset.symbol, sample);
            }
        }
    }

    Ok(())
}

/// Live data task: subscribes to WebSocket streams for all assets.
pub async fn run_live(
    store: Arc<RwLock<DataStore>>,
    config: &Config,
) -> Result<()> {
    let assets = universe();
    let mut ws = if config.use_testnet {
        hypercore::testnet_ws()
    } else {
        hypercore::mainnet_ws()
    };

    // Subscribe to per-asset feeds.
    for asset in &assets {
        ws.subscribe(Subscription::Trades { coin: asset.symbol.clone() });
        ws.subscribe(Subscription::L2Book {
            coin: asset.symbol.clone(),
            n_sig_figs: None,
            mantissa: None,
        });
        // 1m candles for live signal refresh.
        ws.subscribe(Subscription::Candle {
            coin: asset.symbol.clone(),
            interval: "1m".into(),
        });
        ws.subscribe(Subscription::Candle {
            coin: asset.symbol.clone(),
            interval: "5m".into(),
        });
        // Real-time funding/mark price.
        ws.subscribe(Subscription::ActiveAssetCtx { coin: asset.symbol.clone() });
    }

    // Cross-asset mid prices.
    ws.subscribe(Subscription::AllMids { dex: None });

    info!("WebSocket subscriptions active for {} assets", assets.len());

    while let Some(event) = ws.next().await {
        use hypersdk::hypercore::ws::Event;
        match event {
            Event::Connected => info!("WebSocket connected"),
            Event::Disconnected => warn!("WebSocket disconnected – auto-reconnecting"),
            Event::Message(msg) => handle_message(msg, &store, &assets),
        }
    }

    Ok(())
}

fn handle_message(msg: Incoming, store: &Arc<RwLock<DataStore>>, assets: &[Asset]) {
    match msg {
        Incoming::Trades(trades) => {
            let mut s = store.write();
            for trade in &trades {
                // Verify it's in our universe before writing.
                if assets.iter().any(|a| a.symbol == trade.coin) {
                    s.push_trade(&trade.coin, trade);
                }
            }
        }

        Incoming::L2Book(book) => {
            if assets.iter().any(|a| a.symbol == book.coin) {
                store.write().update_book(&book.coin, &book);
            }
        }

        Incoming::Candle(candle) => {
            if assets.iter().any(|a| a.symbol == candle.coin) {
                let mut s = store.write();
                // Look up the last bar's close for the same asset+interval to compute log return.
                let prev_close = s
                    .bars_for(&candle.coin, &candle.interval)
                    .and_then(|b| b.back())
                    .map(|b| b.close);
                let bar = Bar::from_candle(&candle, prev_close);
                s.push_bar(&candle.coin, &candle.interval, bar);
            }
        }

        Incoming::AllMids { mids, .. } => {
            store.write().mids = mids;
        }

        Incoming::ActiveAssetCtx { coin, ctx } => {
            if assets.iter().any(|a| a.symbol == coin) {
                use rust_decimal::prelude::ToPrimitive;
                let snap = AssetContextSnapshot {
                    funding_rate: ctx.funding.to_f64().unwrap_or(0.0),
                    open_interest: ctx.open_interest.to_f64().unwrap_or(0.0),
                    mark_px: ctx.mark_px.and_then(|p| p.to_f64()).unwrap_or(0.0),
                    oracle_px: ctx.oracle_px.and_then(|p| p.to_f64()).unwrap_or(0.0),
                    prev_day_px: ctx.prev_day_px.to_f64().unwrap_or(0.0),
                };
                store.write().asset_ctx.insert(coin, snap);
            }
        }

        _ => {}
    }
}
