//! Feature assembly: collects all raw signals into a unified flat feature row
//! that the ML ensemble and HMM consume.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::{
    assets::Asset,
    data::store::DataStore,
    features::{
        funding::{FundingFeatures, volatility_ratio},
        kalman::KalmanFilter,
        technical::*,
    },
    orderbook::features::OrderBookFeatures,
    stats::cointegration::OuProcess,
};

/// One complete feature row for one asset at one bar.
#[derive(Debug, Clone, Default)]
pub struct FeatureRow {
    pub asset: String,
    pub timestamp_ms: u64,

    // ── Technical ──────────────────────────────────────────────
    pub rsi_14: f64,
    pub rsi_2: f64,
    pub macd_hist: f64,
    pub boll_pos: f64,
    pub atr_norm: f64,
    pub roc_1h: f64,
    pub roc_4h: f64,
    pub roc_24h: f64,
    pub dist_high_20d: f64,
    pub dist_low_20d: f64,
    pub volume_z: f64,
    pub garch_vol: f64,

    // ── Kalman ─────────────────────────────────────────────────
    pub kalman_level: f64,
    pub kalman_velocity: f64,
    pub kalman_level_dev: f64, // (price - kalman_level) / kalman_level

    // ── Microstructure ─────────────────────────────────────────
    pub spread_bps: f64,
    pub depth_ofi: f64,
    pub depth_ratio: f64,
    pub price_impact: f64,
    pub micro_price_dev: f64,
    pub trade_imbalance: f64,
    pub hawkes_cross_ratio: f64,

    // ── Funding ────────────────────────────────────────────────
    pub funding_z: f64,
    pub funding_momentum: f64,
    pub vol_ratio: f64,
    pub premium_pct: f64,

    // ── Cross-asset ────────────────────────────────────────────
    pub beta_to_btc: f64,
    pub idiosyncratic_ret: f64,
    pub ou_z_score: f64,

    // ── Macro ──────────────────────────────────────────────────
    pub corr_to_btc: f64,
}

impl FeatureRow {
    /// Convert to flat f64 vector for ML input. Order must match model training.
    pub fn to_feature_vec(&self) -> Vec<f64> {
        vec![
            self.rsi_14,
            self.rsi_2,
            self.macd_hist,
            self.boll_pos,
            self.atr_norm,
            self.roc_1h,
            self.roc_4h,
            self.roc_24h,
            self.dist_high_20d,
            self.dist_low_20d,
            self.volume_z,
            self.garch_vol,
            self.kalman_level_dev,
            self.kalman_velocity,
            self.spread_bps,
            self.depth_ofi,
            self.depth_ratio,
            self.price_impact,
            self.micro_price_dev,
            self.trade_imbalance,
            self.hawkes_cross_ratio,
            self.funding_z,
            self.funding_momentum,
            self.vol_ratio,
            self.premium_pct,
            self.beta_to_btc,
            self.idiosyncratic_ret,
            self.ou_z_score,
            self.corr_to_btc,
        ]
    }

    pub fn n_features() -> usize {
        29
    }
}

/// Rolling store of recent feature rows per asset.
#[derive(Debug, Default)]
pub struct FeatureStore {
    pub rows: HashMap<String, Vec<FeatureRow>>,
    pub max_rows: usize,
}

impl FeatureStore {
    pub fn new(max_rows: usize) -> Self {
        Self {
            rows: HashMap::new(),
            max_rows,
        }
    }

    pub fn push(&mut self, row: FeatureRow) {
        let v = self.rows.entry(row.asset.clone()).or_default();
        v.push(row);
        if v.len() > self.max_rows {
            v.remove(0);
        }
    }

    pub fn latest(&self, asset: &str) -> Option<&FeatureRow> {
        self.rows.get(asset)?.last()
    }

    /// Feature matrix (n_samples × n_features) for model training.
    pub fn feature_matrix(&self, asset: &str) -> Vec<Vec<f64>> {
        self.rows
            .get(asset)
            .map(|rows| rows.iter().map(|r| r.to_feature_vec()).collect())
            .unwrap_or_default()
    }
}

/// Assemble one feature row from the data store for a given asset.
///
/// `garch_vol`: pre-computed GARCH conditional volatility (from stats module).
/// `ou_z`: pre-computed OU spread z-score (from cointegration module).
/// `btc_returns`: BTC return series for cross-asset features.
pub fn assemble_row(
    asset: &Asset,
    store: &DataStore,
    kalman: &mut KalmanFilter,
    garch_vol: f64,
    ou_z: f64,
    btc_returns: &[f64],
    hawkes_cross_ratio: f64,
) -> Option<FeatureRow> {
    let sym = &asset.symbol;
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_millis() as u64;

    // 5m bars for most indicators (primary signal timeframe).
    let bars_5m = store.bars_for(sym, "5m")?;
    if bars_5m.len() < 30 {
        return None;
    }

    let closes: Vec<f64> = bars_5m.iter().map(|b| b.close).collect();
    let highs: Vec<f64> = bars_5m.iter().map(|b| b.high).collect();
    let lows: Vec<f64> = bars_5m.iter().map(|b| b.low).collect();
    let volumes: Vec<f64> = bars_5m.iter().map(|b| b.volume).collect();
    let returns_5m: Vec<f64> = bars_5m.iter().map(|b| b.log_return).collect();

    // 1h bars for medium-term features.
    let returns_1h: Vec<f64> = store
        .bars_for(sym, "1h")
        .map(|b| b.iter().map(|bar| bar.log_return).collect())
        .unwrap_or_default();

    // 1d bars for long-term features.
    let returns_1d: Vec<f64> = store
        .bars_for(sym, "1d")
        .map(|b| b.iter().map(|bar| bar.log_return).collect())
        .unwrap_or_default();
    let closes_1d: Vec<f64> = store
        .bars_for(sym, "1d")
        .map(|b| b.iter().map(|bar| bar.close).collect())
        .unwrap_or_default();

    let price = *closes.last()?;
    let (kl, kv) = kalman.update(price);
    let kalman_level_dev = if kl.abs() > 1e-10 { (price - kl) / kl } else { 0.0 };

    // RSI (on 5m closes mapped to ~1h equivalent = 12 bars).
    let rsi_14 = rsi(&closes, 14).unwrap_or(50.0);
    let rsi_2 = rsi(&closes, 2).unwrap_or(50.0);

    let (_, _, macd_hist) = macd(&closes).unwrap_or((0.0, 0.0, 0.0));
    let boll_pos = bollinger_position(&closes, 20, 2.0).unwrap_or(0.5);
    let atr_val = atr(&highs, &lows, &closes, 14).unwrap_or(0.0);
    let atr_norm = if price > 1e-10 { atr_val / price } else { 0.0 };

    // ROC in 5m bars: 12×5m ≈ 1h, 48×5m ≈ 4h, 288×5m ≈ 1d.
    let roc_1h = roc(&closes, 12).unwrap_or(0.0);
    let roc_4h = roc(&closes, 48).unwrap_or(0.0);
    let roc_24h = roc(&closes, 288).unwrap_or(0.0);

    // 20-day distance from extremes: use 1d bars (20 bars = 20 days).
    let dist_high_20d = dist_from_rolling_high(&closes_1d, 20).unwrap_or(0.0);
    let dist_low_20d = dist_from_rolling_low(&closes_1d, 20).unwrap_or(0.0);

    let volume_z = volume_z_score(&volumes, 60).unwrap_or(0.0);

    // Order book features.
    let book = store.books.get(sym).cloned().unwrap_or_default();
    let adv_24h_usd = volume_24h_usd(store, sym);
    let ob = OrderBookFeatures::compute(&book, adv_24h_usd);

    let trade_imbalance = store
        .trades
        .get(sym)
        .map(|t| t.imbalance())
        .unwrap_or(0.5);

    // Funding features.
    let ctx = store.asset_ctx.get(sym).cloned().unwrap_or_default();
    let funding_hist = store.funding.get(sym);
    let ff = if let Some(hist) = funding_hist {
        FundingFeatures::compute(hist, ctx.mark_px, ctx.oracle_px)
    } else {
        FundingFeatures::default()
    };

    let vol_ratio = volatility_ratio(&returns_1h, &returns_1d);

    // Cross-asset vs BTC.
    let beta_to_btc = rolling_beta(&returns_5m, btc_returns, 60).unwrap_or(1.0);
    let btc_last_ret = btc_returns.last().copied().unwrap_or(0.0);
    let asset_last_ret = returns_5m.last().copied().unwrap_or(0.0);
    let idiosyncratic_ret = asset_last_ret - beta_to_btc * btc_last_ret;
    let corr_to_btc = rolling_correlation(&returns_5m, btc_returns, 60).unwrap_or(0.0);

    Some(FeatureRow {
        asset: sym.clone(),
        timestamp_ms: now_ms,
        rsi_14,
        rsi_2,
        macd_hist,
        boll_pos,
        atr_norm,
        roc_1h,
        roc_4h,
        roc_24h,
        dist_high_20d,
        dist_low_20d,
        volume_z,
        garch_vol,
        kalman_level: kl,
        kalman_velocity: kv,
        kalman_level_dev,
        spread_bps: ob.spread_bps,
        depth_ofi: ob.depth_ofi,
        depth_ratio: ob.depth_ratio,
        price_impact: ob.price_impact_1pct,
        micro_price_dev: (ob.micro_price - ob.mid_price) / ob.mid_price.max(1e-10),
        trade_imbalance,
        hawkes_cross_ratio,
        funding_z: ff.z_score,
        funding_momentum: ff.momentum,
        vol_ratio,
        premium_pct: ff.premium_pct,
        beta_to_btc,
        idiosyncratic_ret,
        ou_z_score: ou_z,
        corr_to_btc,
    })
}

/// Estimate 24h ADV in USD from 1h bars.
fn volume_24h_usd(store: &DataStore, asset: &str) -> f64 {
    store
        .bars_for(asset, "1h")
        .map(|bars| {
            bars.iter()
                .rev()
                .take(24)
                .map(|b| b.volume * b.close)
                .sum::<f64>()
        })
        .unwrap_or(1_000_000.0)
}
