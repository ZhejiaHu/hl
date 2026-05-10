# quant_trader

Quantitative perpetual-futures trading system for the [Hyperliquid](https://hyperliquid.xyz) DEX, implemented in Rust. The system ingests live order-book and trade data, runs a layered statistical and ML pipeline, detects market regimes, and executes size-risk-managed positions autonomously.

---

## Pipeline overview

```
┌─────────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                     │
│  REST bootstrap (180 days OHLCV) + WebSocket live streams       │
│  Candles · L2 order book · Trades · Funding · Asset context     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STATISTICAL MODELS  (fitted weekly, or at startup)             │
│                                                                 │
│  ┌────────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│  │  GARCH(1,1)-t  │  │  Return dist.    │  │ Cointegration  │  │
│  │  σ²_t forecast │  │  Student-t       │  │ Engle–Granger  │  │
│  │  (per asset,   │  │  Cauchy          │  │ ADF test       │  │
│  │   1h returns)  │  │  Discrete hist.  │  │ OU spread fit  │  │
│  │                │  │  → best AIC wins │  │                │  │
│  └────────────────┘  └──────────────────┘  └────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  FEATURE ENGINEERING  (every 5-min bar)                        │
│                                                                 │
│  Kalman filter  │  Order-book microstructure  │  Funding        │
│  2D state:      │  spread_bps                 │  z-score        │
│  level + vel.   │  depth OFI / ratio          │  momentum       │
│  level_dev      │  price_impact_1pct          │  premium_pct    │
│                 │  micro_price_dev            │  vol_ratio      │
│                 │  trade_imbalance            │                 │
│                 │  Hawkes cross-excitation    │                 │
│                                                                 │
│  Cross-asset (vs BTC): beta · idiosyncratic_ret · correlation  │
│  Cointegration:         OU z-score                             │
│  GARCH output:          garch_vol                              │
│                                                                 │
│  → 18-dimensional FeatureRow per asset per bar                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  ML ENSEMBLE  (one model pair per asset)                       │
│                                                                 │
│  ┌──────────────────────────┐  ┌────────────────────────────┐  │
│  │  GbClassifier (3-class)  │  │  GbRegressor               │  │
│  │  gradient-boosted trees  │  │  gradient-boosted trees    │  │
│  │  target: 4h direction    │  │  target: 4h realised vol   │  │
│  │  (up +1 / flat 0 / dn-1) │  │                            │  │
│  └────────────┬─────────────┘  └──────────────┬─────────────┘  │
│               │                               │                 │
│  p_up, p_flat, p_down                predicted_vol_4h           │
│  directional_edge = p_up - p_down                               │
│  signal_confidence  = 1 - H/H_max  (entropy-based)             │
│  score = |edge| × confidence / vol                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  HMM REGIME DETECTION                                          │
│                                                                 │
│  4-state HMM  (Baum–Welch EM training, online filter_last)     │
│                                                                 │
│  State 0 – Trending Bull   max leverage 5×  size scale 1.0     │
│  State 1 – Trending Bear   max leverage 5×  size scale 1.0     │
│  State 2 – Consolidation   max leverage 2×  size scale 0.5     │
│  State 3 – High-Vol Chaos  max leverage 1×  size scale 0.2     │
│                                                                 │
│  7-dim observation: [edge_btc, edge_eth, vol_agg, corr,        │
│                       exposure, funding_z, n_positions]        │
│  Regime confidence threshold: P(state) > 0.65                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  SIGNAL GENERATION                                             │
│                                                                 │
│  Filters:  confidence > 0.60, |edge| > 0.25                   │
│            Consolidation: OU z-score gate (|z| > 1.5)         │
│            HighVol: no new entries if positions exist          │
│                                                                 │
│  Entry:    Kalman-filtered price (limit order target)          │
│  Stop:     2× predicted_vol_4h from entry                      │
│  Target:   stop × regime.min_rr (2.0 – 3.0 depending on state)│
│                                                                 │
│  Sizing:   Fractional Kelly leverage (× kelly_fraction=0.3)   │
│            Vol-targeting: NAV × lev × target_vol / eff_vol    │
│            eff_vol = max(predicted_vol_4h, |CVaR₉₅(dist)|)    │
│            ← tail-risk floor from fitted return distribution   │
│                                                                 │
│  Ranking:  top-3 assets by score = |edge|×conf/vol            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  RISK MANAGEMENT  (7 pre-trade checks)                         │
│                                                                 │
│  1. Portfolio drawdown < daily_drawdown_limit (8%)             │
│  2. Post-trade portfolio leverage ≤ max_portfolio_leverage (3×)│
│  3. Single-asset weight ≤ max_single_asset_weight (30%)        │
│  4. Cross-asset correlation with existing positions ≤ 75%      │
│  5. Risk-reward ratio ≥ 1.5                                    │
│  6. Open positions ≤ 5                                         │
│  7. Stop distance ≥ |CVaR₉₉(fitted dist)|  [distribution-aware│
│     tail-adequacy: stop must cover the expected tail loss]     │
│                                                                 │
│  Position monitoring:                                          │
│  - Time stop: 48h max holding period                           │
│  - Vol stop: >5% adverse move while in loss                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  EXECUTION  (Hyperliquid via hypersdk)                         │
│                                                                 │
│  Entry:       GTC limit order at Kalman price                  │
│  Stop-loss:   Trigger/Sl reduce-only order                     │
│  Take-profit: Trigger/Tp reduce-only order                     │
│  Exit:        IOC market close + cancel outstanding SL/TP      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module structure

```
src/
├── main.rs                  — async main; bootstrap → live loop → weekly retrain
├── assets.rs                — Asset universe and metadata
├── config.rs                — Config from environment variables
│
├── data/
│   ├── store.rs             — DataStore (shared RwLock state): bars, books, trades, funding
│   └── ingestion.rs         — REST historical bootstrap + WebSocket live subscriptions
│
├── stats/
│   ├── distribution.rs      — StudentT · CauchyDist · DiscreteDist · ReturnDist (AIC selection)
│   ├── garch.rs             — GARCH(1,1)-t: MLE via projected gradient descent
│   └── cointegration.rs     — Engle–Granger test · OuProcess fitting · z-score
│
├── orderbook/
│   ├── features.rs          — OrderBookFeatures: spread, OFI, depth ratio, impact, micro-price
│   └── hawkes.rs            — Bivariate Hawkes process: buy/sell MLE, cross-excitation ratio
│
├── features/
│   ├── kalman.rs            — 2D Kalman filter: state [level, velocity]
│   ├── funding.rs           — FundingFeatures: z-score, momentum, carry, premium, vol_ratio
│   └── assembler.rs         — FeatureRow (18 dims) · FeatureStore · assemble_row()
│
├── ml/
│   ├── tree.rs              — Greedy MSE regression tree (weak learner)
│   ├── gradient_boost.rs    — GbRegressor · GbClassifier (3 one-vs-rest + softmax)
│   └── ensemble.rs          — ModelEnsemble: per-asset classifier + regressor
│
├── hmm/
│   └── model.rs             — HmmModel: Baum–Welch EM · Viterbi · filter_last (online)
│
├── signals/
│   └── generator.rs         — generate_signals(): regime-gated, Kelly-sized TradeSignal
│
├── risk/
│   └── manager.rs           — RiskManager (7 checks) · PortfolioState · Position
│
├── execution/
│   └── executor.rs          — Executor: entry / SL / TP / cancel / close via hypersdk
│
└── error.rs                 — TradingError enum
```

---

## Asset universe

| Symbol | Index | Max lev. | Max weight | Role |
|--------|------:|----------:|-----------:|------|
| BTC    |     0 |       50× |        30% | Liquidity anchor, regime benchmark |
| ETH    |     1 |       50× |        25% | Liquidity anchor, regime benchmark |
| SOL    |     2 |       20× |        15% | Ecosystem momentum |
| HYPE   |   107 |       10× |        10% | Ecosystem momentum |
| ARB    |     8 |       10× |        10% | L2/DeFi beta |
| MATIC  |     6 |       10× |        10% | L2/DeFi beta |
| WIF    |    30 |       10× |         5% | Meme momentum (hard cap) |

---

## Feature vector (18 dimensions)

| # | Field | Source |
|---|-------|--------|
| 1 | `garch_vol` | GARCH(1,1)-t one-step σ forecast |
| 2 | `kalman_level_dev` | (price − level) / level |
| 3 | `kalman_velocity` | Kalman state[1] |
| 4 | `spread_bps` | Best bid/ask spread in basis points |
| 5 | `depth_ofi` | Order-flow imbalance (depth-weighted) |
| 6 | `depth_ratio` | Bid depth / (bid + ask depth) |
| 7 | `price_impact` | Market-impact cost at 1% ADV |
| 8 | `micro_price_dev` | (micro_price − mid) / mid |
| 9 | `trade_imbalance` | Buy volume / total volume (rolling 2h) |
| 10 | `hawkes_cross_ratio` | α_bs / α_bb: buy→sell excitation |
| 11 | `funding_z` | Z-score of funding rate vs 30-sample history |
| 12 | `funding_momentum` | Trend of recent funding samples |
| 13 | `vol_ratio` | Realised vol (1h) / realised vol (1d) |
| 14 | `premium_pct` | (mark − oracle) / oracle |
| 15 | `beta_to_btc` | Rolling 60-bar beta vs BTC |
| 16 | `idiosyncratic_ret` | ret − β × btc_ret |
| 17 | `ou_z_score` | OU spread z-score (cointegration pair) |
| 18 | `corr_to_btc` | Rolling 60-bar Pearson correlation vs BTC |

---

## Return distribution fitting

Three candidate models are fitted to each asset's 1h log-return series and ranked by AIC. The winner is stored in `dist_models` and used downstream for CVaR-aware sizing and stop-adequacy checks.

| Distribution | Params | Fitting method | VaR / CVaR |
|---|---|---|---|
| **Student-t** | μ, σ, ν | EM-MLE (digamma Newton) | Exact quantile + closed-form ES |
| **Cauchy** | x₀, γ | Alternating MLE: IRLS (location) + Newton (scale) | Exact quantile; bounded CVaR (0.001 tail truncation) |
| **Discrete** | k−1 bins | Scott's-rule histogram + Laplace smoothing | Piecewise-linear CDF; bin-midpoint ES |

AIC = 2k − 2·LL. Lower is better. The Student-t wins on symmetric heavy-tailed assets; Cauchy on assets with near-infinite-variance regimes; Discrete on skewed or bimodal distributions.

---

## Configuration

All parameters are read from environment variables (`.env` supported via `dotenvy`):

| Variable | Default | Description |
|---|---|---|
| `PRIVATE_KEY` | — | Hyperliquid wallet private key |
| `WALLET_ADDRESS` | — | On-chain wallet address |
| `USE_TESTNET` | `false` | Route to testnet instead of mainnet |
| `MAX_PORTFOLIO_LEVERAGE` | `3.0` | Hard portfolio leverage cap |
| `MAX_SINGLE_ASSET_WEIGHT` | `0.30` | Max notional weight per asset |
| `DAILY_DRAWDOWN_LIMIT` | `0.08` | Halt trading above 8% drawdown |
| `TARGET_DAILY_VOL` | `0.015` | Vol-targeting denominator (1.5% daily) |
| `KELLY_FRACTION` | `0.30` | Fractional Kelly scaling (30%) |
| `GB_N_ESTIMATORS` | `100` | Gradient-boosted trees per model |
| `GB_LEARNING_RATE` | `0.1` | Shrinkage per tree |
| `GB_MAX_DEPTH` | `3` | Maximum tree depth |
| `HISTORY_DAYS` | `180` | Days of OHLCV to bootstrap at startup |
| `SIGNAL_INTERVAL_SECS` | `300` | Main loop interval (5 minutes) |

---

## Retraining schedule

- **Startup**: full retrain before the first tick.
- **Weekly**: every 2016 ticks (2016 × 5 min = 7 days), triggered automatically in the main loop.

Each retrain fits: GARCH(1,1)-t · return distributions · Engle–Granger cointegration · OU processes · rebuilds the feature store · retrains the ML ensemble per asset.

---

## Dependencies

| Crate | Purpose |
|---|---|
| `hypersdk` | Hyperliquid REST + WebSocket client |
| `tokio` | Async runtime |
| `nalgebra` | Matrix algebra (Kalman filter, HMM) |
| `statrs` | `ln_gamma` for Student-t / GARCH likelihood |
| `parking_lot` | Low-latency `RwLock` for shared state |
| `serde` / `serde_json` | Signal serialisation |
| `uuid` | Unique signal IDs |
| `anyhow` | Error handling |
| `tracing` | Structured logging |
| `dotenvy` | `.env` file loading |
