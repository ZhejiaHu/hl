use std::collections::{HashMap, VecDeque};

use hypersdk::hypercore::types::{BookLevel, Candle, Trade};
use rust_decimal::Decimal;

/// A single OHLCV bar with derived fields.
#[derive(Debug, Clone)]
pub struct Bar {
    pub open_time: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub num_trades: u64,
    pub log_return: f64,
}

impl Bar {
    pub fn from_candle(c: &Candle, prev_close: Option<f64>) -> Self {
        let close = c.close.to_f64().unwrap_or(0.0);
        let log_return = prev_close
            .filter(|&p| p > 0.0)
            .map(|p| (close / p).ln())
            .unwrap_or(0.0);
        Self {
            open_time: c.open_time,
            open: c.open.to_f64().unwrap_or(0.0),
            high: c.high.to_f64().unwrap_or(0.0),
            low: c.low.to_f64().unwrap_or(0.0),
            close,
            volume: c.volume.to_f64().unwrap_or(0.0),
            num_trades: c.num_trades,
            log_return,
        }
    }
}

/// Rolling order book state for one asset.
#[derive(Debug, Clone, Default)]
pub struct OrderBookState {
    pub bids: Vec<BookLevel>,
    pub asks: Vec<BookLevel>,
    pub last_update_ms: u64,
}

impl OrderBookState {
    pub fn best_bid(&self) -> Option<&BookLevel> {
        self.bids.first()
    }

    pub fn best_ask(&self) -> Option<&BookLevel> {
        self.asks.first()
    }

    pub fn mid_price(&self) -> Option<f64> {
        let bid = self.best_bid()?.px.to_f64()?;
        let ask = self.best_ask()?.px.to_f64()?;
        Some((bid + ask) / 2.0)
    }

    pub fn spread_bps(&self) -> Option<f64> {
        let bid = self.best_bid()?.px.to_f64()?;
        let ask = self.best_ask()?.px.to_f64()?;
        let mid = (bid + ask) / 2.0;
        Some((ask - bid) / mid * 10_000.0)
    }

    /// Depth-weighted bid size at top N levels.
    pub fn bid_depth(&self, levels: usize) -> f64 {
        self.bids.iter().take(levels).map(|l| l.sz.to_f64().unwrap_or(0.0)).sum()
    }

    /// Depth-weighted ask size at top N levels.
    pub fn ask_depth(&self, levels: usize) -> f64 {
        self.asks.iter().take(levels).map(|l| l.sz.to_f64().unwrap_or(0.0)).sum()
    }

    /// Estimate market impact of trading `notional` USD worth.
    pub fn price_impact(&self, notional: f64, is_buy: bool) -> f64 {
        let mid = self.mid_price().unwrap_or(0.0);
        if mid == 0.0 {
            return 0.0;
        }
        let levels = if is_buy { &self.asks } else { &self.bids };
        let mut remaining = notional;
        let mut cost = 0.0;
        for level in levels {
            let px = level.px.to_f64().unwrap_or(0.0);
            let sz = level.sz.to_f64().unwrap_or(0.0);
            let level_notional = px * sz;
            if remaining <= level_notional {
                cost += remaining;
                remaining = 0.0;
                break;
            }
            cost += level_notional;
            remaining -= level_notional;
        }
        if remaining > 0.0 {
            return 1.0; // Book fully depleted – extreme impact
        }
        (cost / notional - mid) / mid
    }
}

/// Rolling trade history for Hawkes process estimation.
#[derive(Debug, Clone, Default)]
pub struct TradeHistory {
    pub buy_times_ms: VecDeque<u64>,
    pub sell_times_ms: VecDeque<u64>,
    pub max_window_ms: u64,
}

impl TradeHistory {
    pub fn new(window_ms: u64) -> Self {
        Self {
            buy_times_ms: VecDeque::new(),
            sell_times_ms: VecDeque::new(),
            max_window_ms: window_ms,
        }
    }

    pub fn push(&mut self, trade: &Trade) {
        let t = trade.time;
        match trade.side {
            hypersdk::hypercore::types::Side::Bid => self.buy_times_ms.push_back(t),
            hypersdk::hypercore::types::Side::Ask => self.sell_times_ms.push_back(t),
        }
        self.evict(t);
    }

    fn evict(&mut self, now: u64) {
        let cutoff = now.saturating_sub(self.max_window_ms);
        while self.buy_times_ms.front().is_some_and(|&t| t < cutoff) {
            self.buy_times_ms.pop_front();
        }
        while self.sell_times_ms.front().is_some_and(|&t| t < cutoff) {
            self.sell_times_ms.pop_front();
        }
    }

    /// Buy volume fraction in the window.
    pub fn imbalance(&self) -> f64 {
        let total = self.buy_times_ms.len() + self.sell_times_ms.len();
        if total == 0 {
            return 0.5;
        }
        self.buy_times_ms.len() as f64 / total as f64
    }
}

/// Funding rate sample for one asset.
#[derive(Debug, Clone)]
pub struct FundingSample {
    pub time: u64,
    pub rate: f64,
    pub premium: f64,
}

/// Central shared data store — all live data lands here.
#[derive(Debug, Default)]
pub struct DataStore {
    /// OHLCV bars per (asset, interval_str). VecDeque for bounded rolling window.
    pub bars: HashMap<(String, String), VecDeque<Bar>>,

    /// Current order book state per asset.
    pub books: HashMap<String, OrderBookState>,

    /// Rolling trade history per asset (2h window for Hawkes estimation).
    pub trades: HashMap<String, TradeHistory>,

    /// Funding rate history per asset.
    pub funding: HashMap<String, VecDeque<FundingSample>>,

    /// Latest all-mids snapshot.
    pub mids: HashMap<String, Decimal>,

    /// Latest asset context (funding rate, open interest, mark price).
    pub asset_ctx: HashMap<String, AssetContextSnapshot>,

    /// Maximum bars to keep in memory per (asset, interval).
    pub max_bars: usize,
}

#[derive(Debug, Clone, Default)]
pub struct AssetContextSnapshot {
    pub funding_rate: f64,
    pub open_interest: f64,
    pub mark_px: f64,
    pub oracle_px: f64,
    pub prev_day_px: f64,
}

impl DataStore {
    pub fn new(max_bars: usize) -> Self {
        Self {
            max_bars,
            ..Default::default()
        }
    }

    pub fn push_bar(&mut self, asset: &str, interval: &str, bar: Bar) {
        let key = (asset.to_string(), interval.to_string());
        let deque = self.bars.entry(key).or_default();
        deque.push_back(bar);
        if deque.len() > self.max_bars {
            deque.pop_front();
        }
    }

    pub fn push_trade(&mut self, asset: &str, trade: &Trade) {
        self.trades
            .entry(asset.to_string())
            .or_insert_with(|| TradeHistory::new(2 * 3_600_000)) // 2h
            .push(trade);
    }

    pub fn update_book(&mut self, asset: &str, book: &hypersdk::hypercore::types::L2Book) {
        let state = self.books.entry(asset.to_string()).or_default();
        state.bids = book.levels[1].clone();
        state.asks = book.levels[0].clone();
        state.last_update_ms = book.time;
    }

    pub fn push_funding(&mut self, asset: &str, sample: FundingSample) {
        let deque = self.funding.entry(asset.to_string()).or_default();
        deque.push_back(sample);
        if deque.len() > 200 {
            deque.pop_front();
        }
    }

    pub fn bars_for(&self, asset: &str, interval: &str) -> Option<&VecDeque<Bar>> {
        self.bars.get(&(asset.to_string(), interval.to_string()))
    }

    pub fn closes(&self, asset: &str, interval: &str) -> Vec<f64> {
        self.bars_for(asset, interval)
            .map(|b| b.iter().map(|bar| bar.close).collect())
            .unwrap_or_default()
    }

    pub fn returns(&self, asset: &str, interval: &str) -> Vec<f64> {
        self.bars_for(asset, interval)
            .map(|b| b.iter().map(|bar| bar.log_return).collect())
            .unwrap_or_default()
    }
}

trait ToF64 {
    fn to_f64(&self) -> Option<f64>;
}
impl ToF64 for Decimal {
    fn to_f64(&self) -> Option<f64> {
        rust_decimal::prelude::ToPrimitive::to_f64(self)
    }
}
