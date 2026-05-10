//! Microstructure features derived from the limit order book.
//!
//! All features are dimensionless (normalised by mid-price or total volume)
//! so they are directly comparable across assets and stable as inputs to ML models.

use crate::data::store::OrderBookState;

/// Full set of order book features for one asset at one point in time.
#[derive(Debug, Clone, Default)]
pub struct OrderBookFeatures {
    /// Mid price.
    pub mid_price: f64,
    /// Bid–ask spread in basis points.
    pub spread_bps: f64,
    /// Order flow imbalance in [-1, 1]:  (bid_depth - ask_depth) / total_depth.
    pub depth_ofi: f64,
    /// Top-5 bid depth (in quote units).
    pub bid_depth_5: f64,
    /// Top-5 ask depth (in quote units).
    pub ask_depth_5: f64,
    /// Depth ratio: bid_depth_5 / ask_depth_5.
    pub depth_ratio: f64,
    /// Estimated price impact of trading 1% of 24h ADV (fraction of mid).
    pub price_impact_1pct: f64,
    /// Volume-weighted mid price (microprice).
    pub micro_price: f64,
}

impl OrderBookFeatures {
    /// Compute all microstructure features from the current book state.
    pub fn compute(book: &OrderBookState, adv_24h_usd: f64) -> Self {
        let mid = book.mid_price().unwrap_or(0.0);
        if mid == 0.0 {
            return Self::default();
        }

        let spread_bps = book.spread_bps().unwrap_or(0.0);
        let bid5 = book.bid_depth(5);
        let ask5 = book.ask_depth(5);
        let total = bid5 + ask5;
        let depth_ofi = if total > 0.0 { (bid5 - ask5) / total } else { 0.0 };
        let depth_ratio = if ask5 > 0.0 { bid5 / ask5 } else { 1.0 };

        // Microprice (volume-weighted mid).
        let micro_price = if let (Some(bid), Some(ask)) = (book.best_bid(), book.best_ask()) {
            let b_sz = bid.sz.to_f64();
            let a_sz = ask.sz.to_f64();
            let b_px = bid.px.to_f64().unwrap_or(0.0);
            let a_px = ask.px.to_f64().unwrap_or(0.0);
            let total_sz = b_sz.unwrap_or(0.0) + a_sz.unwrap_or(0.0);
            if total_sz > 0.0 {
                (b_px * a_sz.unwrap_or(0.0) + a_px * b_sz.unwrap_or(0.0)) / total_sz
            } else {
                mid
            }
        } else {
            mid
        };

        // Price impact at 1% of ADV.
        let impact_notional = adv_24h_usd * 0.01;
        let price_impact_buy = book.price_impact(impact_notional, true);
        let price_impact_1pct = price_impact_buy.abs();

        Self {
            mid_price: mid,
            spread_bps,
            depth_ofi,
            bid_depth_5: bid5,
            ask_depth_5: ask5,
            depth_ratio,
            price_impact_1pct,
            micro_price,
        }
    }

    /// Convert to a flat f64 vector for use in ML models.
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.spread_bps,
            self.depth_ofi,
            self.depth_ratio,
            self.price_impact_1pct,
            (self.micro_price - self.mid_price) / self.mid_price.max(1e-10),
        ]
    }
}

trait ToF64Opt {
    fn to_f64(&self) -> Option<f64>;
}

impl ToF64Opt for rust_decimal::Decimal {
    fn to_f64(&self) -> Option<f64> {
        rust_decimal::prelude::ToPrimitive::to_f64(self)
    }
}
