//! Feature assembly: collects all raw signals into a unified flat feature row
//! that the ML ensemble and HMM consume.

use std::collections::HashMap;

use crate::{
    assets::Asset,
    data::store::DataStore,
    features::{
        funding::{FundingFeatures, volatility_ratio},
        kalman::KalmanFilter,
    },
    orderbook::features::OrderBookFeatures,
};

/// One complete feature row for one asset at one bar.
#[derive(Debug, Clone, Default)]
pub struct FeatureRow {
    pub asset: String,
    pub timestamp_ms: u64,

    // ── GARCH ──────────────────────────────────────────────────
    pub garch_vol: f64,

    // ── Kalman ─────────────────────────────────────────────────
    pub kalman_level: f64,
    pub kalman_velocity: f64,
    /// (price - kalman_level) / kalman_level
    pub kalman_level_dev: f64,

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
    pub corr_to_btc: f64,
}

impl FeatureRow {
    /// Flat f64 vector for ML input. Order must match model training.
    pub fn to_feature_vec(&self) -> Vec<f64> {
        vec![
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
        18
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
/// - `garch_vol`: pre-computed GARCH conditional vol (from stats module).
/// - `ou_z`: pre-computed OU spread z-score (from cointegration module).
/// - `btc_returns`: BTC 5m log-return series for cross-asset features.
/// - `hawkes_cross_ratio`: buy→sell cross-excitation ratio (from Hawkes module).
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

    // 5m bars — primary timeframe.
    let bars_5m = store.bars_for(sym, "5m")?;
    if bars_5m.len() < 30 {
        return None;
    }

    let closes: Vec<f64> = bars_5m.iter().map(|b| b.close).collect();
    let returns_5m: Vec<f64> = bars_5m.iter().map(|b| b.log_return).collect();

    // 1h and 1d returns for volatility ratio.
    let returns_1h: Vec<f64> = store
        .bars_for(sym, "1h")
        .map(|b| b.iter().map(|bar| bar.log_return).collect())
        .unwrap_or_default();
    let returns_1d: Vec<f64> = store
        .bars_for(sym, "1d")
        .map(|b| b.iter().map(|bar| bar.log_return).collect())
        .unwrap_or_default();

    // Kalman level and velocity.
    let price = *closes.last()?;
    let (kl, kv) = kalman.update(price);
    let kalman_level_dev = if kl.abs() > 1e-10 { (price - kl) / kl } else { 0.0 };

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
    let ff = store.funding.get(sym).map_or_else(FundingFeatures::default, |hist| {
        FundingFeatures::compute(hist, ctx.mark_px, ctx.oracle_px)
    });

    let vol_ratio = volatility_ratio(&returns_1h, &returns_1d);

    // Cross-asset features vs BTC.
    let beta_to_btc = rolling_beta(&returns_5m, btc_returns, 60).unwrap_or(1.0);
    let btc_last_ret = btc_returns.last().copied().unwrap_or(0.0);
    let asset_last_ret = returns_5m.last().copied().unwrap_or(0.0);
    let idiosyncratic_ret = asset_last_ret - beta_to_btc * btc_last_ret;
    let corr_to_btc = rolling_correlation(&returns_5m, btc_returns, 60).unwrap_or(0.0);

    Some(FeatureRow {
        asset: sym.clone(),
        timestamp_ms: now_ms,
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

// ─── private helpers ────────────────────────────────────────────────────────

fn volume_24h_usd(store: &DataStore, asset: &str) -> f64 {
    store
        .bars_for(asset, "1h")
        .map(|bars| bars.iter().rev().take(24).map(|b| b.volume * b.close).sum::<f64>())
        .unwrap_or(1_000_000.0)
}

fn rolling_beta(asset_returns: &[f64], benchmark_returns: &[f64], period: usize) -> Option<f64> {
    let n = asset_returns.len().min(benchmark_returns.len());
    if n < period {
        return None;
    }
    let ra = &asset_returns[n - period..];
    let rb = &benchmark_returns[n - period..];
    let mu_a = ra.iter().sum::<f64>() / period as f64;
    let mu_b = rb.iter().sum::<f64>() / period as f64;
    let cov: f64 = ra.iter().zip(rb).map(|(&a, &b)| (a - mu_a) * (b - mu_b)).sum();
    let var_b: f64 = rb.iter().map(|&b| (b - mu_b).powi(2)).sum();
    if var_b < 1e-15 {
        return Some(0.0);
    }
    Some(cov / var_b)
}

fn rolling_correlation(r1: &[f64], r2: &[f64], period: usize) -> Option<f64> {
    let n = r1.len().min(r2.len());
    if n < period {
        return None;
    }
    let t1 = &r1[n - period..];
    let t2 = &r2[n - period..];
    let mu1 = t1.iter().sum::<f64>() / period as f64;
    let mu2 = t2.iter().sum::<f64>() / period as f64;
    let cov: f64 = t1.iter().zip(t2).map(|(&x, &y)| (x - mu1) * (y - mu2)).sum();
    let var1: f64 = t1.iter().map(|&x| (x - mu1).powi(2)).sum();
    let var2: f64 = t2.iter().map(|&x| (x - mu2).powi(2)).sum();
    let denom = (var1 * var2).sqrt();
    if denom < 1e-15 {
        return Some(0.0);
    }
    Some(cov / denom)
}
