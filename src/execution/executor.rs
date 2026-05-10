//! Order execution: places, monitors, and cancels orders on Hyperliquid.
//!
//! Each signal results in:
//!   1. A limit entry order at the Kalman-filtered mid-price.
//!   2. After fill: a reduce-only stop-loss trigger order.
//!   3. After fill: a reduce-only take-profit limit order.

use std::sync::Arc;

use anyhow::{Result, anyhow};
use hypersdk::hypercore::{
    self,
    NonceHandler,
    PrivateKeySigner,
    types::{
        BatchCancel, BatchOrder, Cancel, OrderGrouping, OrderRequest, OrderResponseStatus,
        OrderTypePlacement, TimeInForce, TpSl,
    },
};
use parking_lot::RwLock;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use tracing::{error, info, warn};

use crate::{
    config::Config,
    data::store::DataStore,
    risk::manager::{PortfolioState, Position},
    signals::generator::{Direction, TradeSignal},
};

pub struct Executor {
    client: hypersdk::hypercore::HttpClient,
    signer: PrivateKeySigner,
    nonce: NonceHandler,
}

impl Executor {
    pub fn new(config: &Config) -> Result<Self> {
        let signer: PrivateKeySigner = config
            .private_key
            .parse()
            .map_err(|e| anyhow!("Invalid private key: {}", e))?;

        let client = if config.use_testnet {
            hypercore::testnet()
        } else {
            hypercore::mainnet()
        };

        Ok(Self {
            client,
            signer,
            nonce: NonceHandler::default(),
        })
    }

    /// Place a limit entry order for `signal`.
    ///
    /// Returns the exchange order ID on success.
    pub async fn place_entry(&self, signal: &TradeSignal) -> Result<u64> {
        let is_buy = signal.direction.is_long();
        let entry_px = decimal_from_f64(signal.entry_price)?;
        let size_per_contract = signal.position_size_usd * signal.leverage / signal.entry_price;
        let sz = decimal_from_f64(size_per_contract)?;

        let order = BatchOrder {
            orders: vec![OrderRequest {
                asset: signal.asset_index,
                is_buy,
                limit_px: entry_px,
                sz,
                reduce_only: false,
                order_type: OrderTypePlacement::Limit {
                    tif: TimeInForce::Gtc,
                },
                cloid: Default::default(),
            }],
            grouping: OrderGrouping::Na,
        };

        let nonce = self.nonce.next();
        let results = self
            .client
            .place(&self.signer, order, nonce, None, None)
            .await
            .map_err(|e| anyhow!("Place order error: {:?}", e))?;

        extract_oid(&results).ok_or_else(|| anyhow!("No OID returned for entry order"))
    }

    /// Place a stop-loss trigger order (reduce-only).
    pub async fn place_stop_loss(&self, signal: &TradeSignal) -> Result<u64> {
        let is_buy = !signal.direction.is_long(); // Opposite direction to close.
        let sl_px = decimal_from_f64(signal.stop_loss)?;
        // Trigger price = sl_px (market order on trigger).
        let size_per_contract = signal.position_size_usd * signal.leverage / signal.entry_price;
        let sz = decimal_from_f64(size_per_contract)?;

        let order = BatchOrder {
            orders: vec![OrderRequest {
                asset: signal.asset_index,
                is_buy,
                limit_px: sl_px,
                sz,
                reduce_only: true,
                order_type: OrderTypePlacement::Trigger {
                    is_market: true,
                    trigger_px: sl_px,
                    tpsl: TpSl::Sl,
                },
                cloid: Default::default(),
            }],
            grouping: OrderGrouping::Na,
        };

        let nonce = self.nonce.next();
        let results = self
            .client
            .place(&self.signer, order, nonce, None, None)
            .await
            .map_err(|e| anyhow!("Stop-loss order error: {:?}", e))?;

        extract_oid(&results).ok_or_else(|| anyhow!("No OID returned for stop-loss order"))
    }

    /// Place a take-profit limit order (reduce-only).
    pub async fn place_take_profit(&self, signal: &TradeSignal) -> Result<u64> {
        let is_buy = !signal.direction.is_long();
        let tp_px = decimal_from_f64(signal.take_profit)?;
        let size_per_contract = signal.position_size_usd * signal.leverage / signal.entry_price;
        let sz = decimal_from_f64(size_per_contract)?;

        let order = BatchOrder {
            orders: vec![OrderRequest {
                asset: signal.asset_index,
                is_buy,
                limit_px: tp_px,
                sz,
                reduce_only: true,
                order_type: OrderTypePlacement::Trigger {
                    is_market: false,
                    trigger_px: tp_px,
                    tpsl: TpSl::Tp,
                },
                cloid: Default::default(),
            }],
            grouping: OrderGrouping::Na,
        };

        let nonce = self.nonce.next();
        let results = self
            .client
            .place(&self.signer, order, nonce, None, None)
            .await
            .map_err(|e| anyhow!("Take-profit order error: {:?}", e))?;

        extract_oid(&results).ok_or_else(|| anyhow!("No OID returned for take-profit order"))
    }

    /// Cancel an order by its exchange-assigned OID.
    pub async fn cancel_order(&self, asset_index: usize, oid: u64) -> Result<()> {
        let batch = BatchCancel {
            cancels: vec![Cancel {
                asset: asset_index,
                oid,
            }],
        };
        let nonce = self.nonce.next();
        self.client
            .cancel(&self.signer, batch, nonce, None, None)
            .await
            .map_err(|e| anyhow!("Cancel error: {:?}", e))?;
        Ok(())
    }

    /// Close a position at market by placing an aggressive limit order.
    /// Uses the opposite direction at the current best bid/ask.
    pub async fn close_position(
        &self,
        pos: &Position,
        store: &Arc<RwLock<DataStore>>,
    ) -> Result<()> {
        let book = {
            let s = store.read();
            s.books.get(&pos.asset).cloned()
        };

        let close_px = if let Some(b) = book {
            match pos.direction {
                Direction::Long => b.best_ask().and_then(|a| {
                    rust_decimal::prelude::ToPrimitive::to_f64(&a.px)
                }),
                Direction::Short => b.best_bid().and_then(|a| {
                    rust_decimal::prelude::ToPrimitive::to_f64(&a.px)
                }),
            }
        } else {
            None
        };

        let close_px = close_px.unwrap_or(pos.entry_price * 1.01);

        let is_buy = !pos.direction.is_long();
        let px = decimal_from_f64(close_px)?;
        let sz = decimal_from_f64(pos.size_usd * pos.leverage / pos.entry_price)?;

        // Determine asset index from known assets.
        let asset_index = crate::assets::universe()
            .iter()
            .find(|a| a.symbol == pos.asset)
            .map(|a| a.index)
            .unwrap_or(0);

        let order = BatchOrder {
            orders: vec![OrderRequest {
                asset: asset_index,
                is_buy,
                limit_px: px,
                sz,
                reduce_only: true,
                order_type: OrderTypePlacement::Limit {
                    tif: TimeInForce::Ioc,
                },
                cloid: Default::default(),
            }],
            grouping: OrderGrouping::Na,
        };

        let nonce = self.nonce.next();
        self.client
            .place(&self.signer, order, nonce, None, None)
            .await
            .map_err(|e| anyhow!("Close position error: {:?}", e))?;

        info!("Closed {} {} position", pos.direction, pos.asset);
        Ok(())
    }

    /// Fetch account state and update portfolio NAV.
    pub async fn refresh_portfolio(
        &self,
        portfolio: &mut PortfolioState,
        wallet: &hypersdk::Address,
    ) -> Result<()> {
        let state = self
            .client
            .clearinghouse_state(*wallet, None)
            .await
            .map_err(|e| anyhow!("clearinghouse_state error: {:?}", e))?;

        use rust_decimal::prelude::ToPrimitive;
        let account_value = state
            .margin_summary
            .account_value
            .to_f64()
            .unwrap_or(portfolio.nav);
        portfolio.update_nav(account_value);
        Ok(())
    }
}

// ─── helpers ────────────────────────────────────────────────────────────────

fn decimal_from_f64(val: f64) -> Result<Decimal> {
    Decimal::try_from(val).map_err(|e| anyhow!("Decimal conversion error: {}", e))
}

fn extract_oid(statuses: &[OrderResponseStatus]) -> Option<u64> {
    for s in statuses {
        match s {
            OrderResponseStatus::Resting { oid, .. } | OrderResponseStatus::Filled { oid, .. } => {
                return Some(*oid);
            }
            _ => {}
        }
    }
    None
}
