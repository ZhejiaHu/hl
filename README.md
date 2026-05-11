# quant_trader

Quantitative perpetual-futures trading system for the [Hyperliquid](https://hyperliquid.xyz) DEX, implemented in Rust. The system ingests live order-book and trade data, runs a fully online probabilistic pipeline, and executes size-risk-managed positions autonomously.

All estimation is online (O(1) per tick). There are no batch-fitted statistical models and no periodic retrain cycles. Regime representation is continuous: the fused probability distribution over future returns **is** the regime.

---

## Pipeline overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                         │
│  REST bootstrap (180 days OHLCV) + WebSocket live streams           │
│  Candles · L2 order book · Trades · Funding · Asset context         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ONLINE ESTIMATION  (every tick, O(1) per update)                   │
│                                                                     │
│  ┌───────────────────────────┐  ┌───────────────────────────────┐  │
│  │  Dual Kalman Filter       │  │  Frequency Extractor          │  │
│  │                           │  │  (Goertzel DFT, causal)       │  │
│  │  Outer KF: parameter track│  │                               │  │
│  │    θ = [μ, κ, log σ_v]    │  │  Periods: 4,8,16,32,64 bars  │  │
│  │    random-walk model      │  │  band_powers[0..4]            │  │
│  │                           │  │  dominant_period              │  │
│  │  Inner KF: state track    │  │  spectral_entropy             │  │
│  │    x = [level, velocity]  │  │  noise_trend_ratio            │  │
│  │    discrete-time OU       │  │  dominant_phase               │  │
│  │                           │  │                               │  │
│  │  Joseph-form P update     │  │  Circular buffer; no          │  │
│  │  Finite-diff Jacobian link│  │  look-ahead                   │  │
│  └───────────────────────────┘  └───────────────────────────────┘  │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Bivariate Hawkes Process  (order-book dynamics)              │  │
│  │  Buy/sell MLE · cross-excitation ratio α_bs/α_bb             │  │
│  └───────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PROBABILISTIC ENSEMBLE  (Bayesian model combination)               │
│                                                                     │
│  ┌───────────────┐  ┌──────────────────┐  ┌──────────────────────┐ │
│  │  Kalman       │  │  Frequency       │  │  Order-book / Hawkes │ │
│  │  signal       │  │  signal          │  │  signal              │ │
│  │  ProbSignal   │  │  ProbSignal      │  │  ProbSignal          │ │
│  └───────┬───────┘  └────────┬─────────┘  └──────────┬───────────┘ │
│          └──────────────────►│◄──────────────────────┘             │
│                              ▼                                      │
│  BayesianFusion: weights ∝ exp(EMA[log p(z_t|model)])              │
│  Mixture predictive mean + variance (between-model uncertainty)     │
│  → EnsembleDistribution { p_up, p_down, predictive_mean/std,       │
│                           confidence, per-source weights }          │
│                                                                     │
│  regime_description(): "high_vol" | "trending" |                   │
│                         "consolidating" | "mixed"  (continuous)     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ASSET DISTRIBUTION  (per-asset pre-optimization bundle)            │
│                                                                     │
│  AssetDistribution {                                                │
│    ensemble: EnsembleDistribution   ← fused regime distribution    │
│    current_fraction: f64            ← signed position / NAV        │
│    entry_price: f64                 ← Kalman level                 │
│    estimated_tc_bps: f64            ← spread + impact proxy        │
│  }                                                                  │
│  Built per-asset per-tick; the joint optimizer's sole input type.   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  JOINT COMBO OPTIMIZER  (KellyOptimizer::optimize_combo)            │
│                                                                     │
│  1. Per-asset Kelly fractions: f*_i = kelly_frac × μ_i/(σ_i²+μ_i²)│
│     TC-adjusted; turnover penalty applied using asset-specific bps  │
│                                                                     │
│  2. Subset selection (cardinality + TC budget constraints):         │
│     objective = Σ_i [ELG_i − Δf_i·tc_i]  (net expected log growth)│
│     constraint: card({i: |f_i|>0}) ≤ MAX_COMBO_ASSETS              │
│     constraint: Σ_i [Δf_i·tc_bps_i] ≤ MAX_COMBO_TC_BPS            │
│     • Exhaustive C(n,k) search when n_candidates ≤ 10              │
│       (ENUMERATE_COMBO_SUBSETS=true; global optimum guaranteed)     │
│     • Greedy rank-by-score for larger candidate sets                │
│                                                                     │
│  3. Portfolio constraint scaling:                                   │
│     Σ|f_i| ≤ max_gross_exposure                                     │
│     |Σ f_i| ≤ max_net_exposure                                      │
│     → uniform scale-down if either is exceeded                      │
│                                                                     │
│  4. Leg construction:                                               │
│     stop distance = 2 × predictive_std (online estimate)           │
│                                                                     │
│  → ComboOrder { legs: Vec<OrderLeg>, total_elg, total_tc_bps,      │
│                 gross_exposure_fraction, net_exposure_fraction }    │
│                                                                     │
│  Example output: "long 0.20 BTC + short 0.15 ETH" (2 legs)         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  RISK MANAGEMENT  (pre-trade checks + kill switches)                │
│                                                                     │
│  Pre-trade (all thresholds from env vars):                          │
│    1. Portfolio drawdown < daily_drawdown_limit                     │
│    2. Post-trade leverage ≤ max_portfolio_leverage                  │
│    3. Single-asset weight ≤ max_single_asset_weight                 │
│    4. Cross-asset correlation ≤ max_correlation (data-driven)       │
│    5. Risk-reward ratio ≥ min_rr                                    │
│    6. Open positions ≤ max_positions                                │
│    7. Signal confidence ≥ min_signal_confidence                     │
│                                                                     │
│  Kill switches (priority order):                                    │
│    P1 CloseAll    — drawdown > hard_drawdown OR leverage > hard_lev │
│    P2 ReduceAll   — vol spike > vol_spike_threshold                  │
│       (fraction)    OR estimator instability flag                   │
│    P3 HaltNewTrades — daily dd / leverage soft breach               │
│                                                                     │
│  Position monitoring:                                               │
│    Time stop: configurable max_hold_ms                              │
│    Vol stop: >5% adverse move while in loss                         │
│    EWMA vol estimator: decay=0.94 (RiskMetrics-style)              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  EXECUTION MODEL  (Almgren-Chriss + fill probability)               │
│                                                                     │
│  Latency check: signal age > max_signal_age_ms → reject             │
│  Spread cost: bid/ask half-spread                                   │
│  Permanent impact: η × (size / ADV)                                │
│  Temporary impact: κ × σ × √(size / ADV)                           │
│  Fill probability: logistic(book_depth / size − 1)                 │
│                                                                     │
│  Executable if: edge > total_cost AND fill_prob > 0.30             │
│  → ExecutionMetrics { adjusted_entry_price, total_cost_pct,        │
│                        fill_probability, is_executable }            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  EXECUTION  (Hyperliquid via hypersdk)                              │
│                                                                     │
│  Entry:       GTC limit order at Kalman-filtered price              │
│  Stop-loss:   Trigger/Sl reduce-only order                          │
│  Take-profit: Trigger/Tp reduce-only order                          │
│  Exit:        IOC market close + cancel outstanding SL/TP           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Module structure

```
src/
├── main.rs                   — async main; bootstrap → live loop
├── assets.rs                 — Asset universe and metadata
├── config.rs                 — Config from environment variables
├── error.rs                  — TradingError enum
│
├── data/
│   ├── store.rs              — DataStore (shared RwLock state): bars, books, trades, funding
│   └── ingestion.rs          — REST historical bootstrap + WebSocket live subscriptions
│
├── estimation/
│   ├── mod.rs                — StateEstimator / ParameterEstimator traits; Observation, StatePosterior
│   └── dual_kalman.rs        — DualKalmanFilter: joint online state + parameter estimation
│
├── orderbook/
│   ├── features.rs           — OrderBookFeatures: spread, OFI, depth ratio, impact, micro-price
│   └── hawkes.rs             — Bivariate Hawkes process: buy/sell MLE, cross-excitation ratio
│
├── features/
│   └── frequency.rs          — FrequencyExtractor (Goertzel DFT); FrequencyFeatures (9 dims)
│
├── ensemble/
│   ├── mod.rs                — ProbabilisticSignal trait; SignalOutput; EnsembleDistribution
│   └── fusion.rs             — BayesianFusion: log-evidence EMA weights; signal adapters
│
├── optimizer/
│   ├── mod.rs                — AssetDistribution; ComboConfig; ComboOrder; OrderLeg; PortfolioOptimizer trait
│   └── kelly.rs              — KellyOptimizer: joint combo + subset selection
│
├── signals/
│   └── generator.rs          — generate_combo() → ComboOrder
│
├── risk/
│   ├── manager.rs            — RiskManager (7 pre-trade checks) · PortfolioState · Position
│   └── limits.rs             — RiskLimits (all env-configurable) · KillSwitchEvaluator · EwmaVolEstimator
│
├── execution/
│   ├── executor.rs           — Executor: entry / SL / TP / cancel / close via hypersdk
│   └── model.rs              — ExecutionModel: Almgren-Chriss impact · latency · fill probability
│
└── backtest/
    └── mod.rs                — BacktestEngine (walk-forward) · BacktestResult · FoldResult
```

---

## Dual Kalman Filter

The `DualKalmanFilter` runs two coupled Kalman filters per tick:

**Outer filter (parameter tracking):** Treats `θ = [μ, κ, log σ_v]` as a random walk. Its "observation" is the innovation of the inner filter, projected via a finite-difference Jacobian `∂innovation/∂θ`. This gives an online estimate of drift, mean-reversion speed, and log-volatility without batch EM.

**Inner filter (state tracking):** Tracks `x = [level, velocity]` using discrete-time OU dynamics conditioned on the current `θ`. Covariance updated via the Joseph form `(I-KH)P(I-KH)ᵀ + K σ²_obs Kᵀ` for numerical stability.

Parameters are clamped each tick: μ∈[−0.01, 0.01], κ∈[0.001, 0.98], log σ_v∈[−12, −1].

---

## Bayesian Ensemble Fusion

`BayesianFusion` combines three `ProbabilisticSignal` sources — DualKF, FrequencyExtractor, Hawkes:

1. Each source emits `log_evidence = log p(z_t | model)` per tick.
2. An EMA of log-evidence is maintained per source: `ema ← decay·ema + (1−decay)·le`.
3. Weights = softmax(ema_vector) with a minimum-weight floor (`fusion_min_weight`) to preserve model diversity.
4. Missing sources (None) are skipped; their weight is redistributed.
5. Mixture predictive variance accounts for both within-model uncertainty (σ²_i) and between-model disagreement ((μ_i − μ_mix)²).

Three built-in adapters convert module outputs into `SignalOutput`:
- `kalman_signal(posterior)` — uses KF innovation and uncertainty
- `frequency_signal(freq, velocity)` — uses dominant-period power and spectral entropy
- `orderbook_signal(hawkes)` — uses cross-excitation ratio as directional pressure

---

## Joint combo optimization

`AssetDistribution` is the boundary type between estimation and optimization. After each tick, the main loop builds one per asset:

```
DualKF posterior  → entry_price, predictive_std (via BayesianFusion)
BayesianFusion    → ensemble: EnsembleDistribution
order book spread → estimated_tc_bps
portfolio state   → current_fraction (signed)
```

`KellyOptimizer::optimize_combo` then solves the joint problem in four steps:

| Step | What happens |
|---|---|
| 1 — Per-asset sizing | f*_i = kelly_fraction × μ_i/(σ_i²+μ_i²), TC-adjusted per asset |
| 2 — Subset selection | Maximise Σ(ELG_i − Δf_i·tc_i) subject to cardinality and TC budget |
| 3 — Exposure scaling | Uniform scale-down if gross or net exposure exceeds portfolio limits |
| 4 — Leg construction | Entry, stop (2 × predictive_std), take-profit (regime-dependent R:R) |

**Subset selection** uses exhaustive C(n,k) enumeration when n_candidates ≤ 10 (at most C(10,3)=120 evaluations; global optimum guaranteed) and falls back to greedy ranking for larger sets.

**TC budget constraint**: Σ_i [|f_i_new − f_i_prev| × estimated_tc_bps_i] ≤ MAX_COMBO_TC_BPS ensures the total round-trip cost of the combo stays within the configured budget.

---

## Walk-forward Backtest

```rust
let engine = BacktestEngine::new(WalkForwardConfig {
    train_bars: 500,
    test_bars: 100,
    max_folds: 0,        // 0 = use all available data
    initial_nav: 100_000.0,
    simulate_execution: true,
    adv_24h_usd: 50_000_000.0,
});
let result = engine.run(&bars).await?;
println!("{:#?}", result.aggregate());
```

Each fold:
1. **Training phase** — warm up `DualKalmanFilter`, `FrequencyExtractor`, `BayesianFusion` on `train_bars` bars. Estimators are reset at the start of each fold to prevent leakage.
2. **Test phase** — simulate trades on `test_bars` bars using the same pipeline as the live system.

Reported metrics per fold and aggregated: Sharpe, Sortino, Calmar, max drawdown (value + duration), hit rate, annualised turnover.

---

## Kill switches

`KillSwitchEvaluator` returns the highest-priority active action:

| Priority | Action | Trigger |
|---|---|---|
| 1 (highest) | `CloseAll` | drawdown > `hard_drawdown_limit` **or** leverage > `hard_leverage_limit` |
| 2 | `ReduceAll { fraction }` | EWMA vol > `vol_spike_threshold` **or** estimator instability |
| 3 | `HaltNewTrades` | daily drawdown > `daily_drawdown_limit` **or** leverage > `max_portfolio_leverage` |

If `CloseAll` is triggered, all open positions are liquidated via IOC market orders before the next tick.

---

## Configuration

All parameters are read from environment variables (`.env` supported via `dotenvy`):

| Variable | Default | Description |
|---|---|---|
| `PRIVATE_KEY` | — | Hyperliquid wallet private key |
| `WALLET_ADDRESS` | — | On-chain wallet address |
| `USE_TESTNET` | `false` | Route to testnet |
| `MAX_PORTFOLIO_LEVERAGE` | `3.0` | Soft portfolio leverage cap |
| `MAX_SINGLE_ASSET_WEIGHT` | `0.30` | Max notional weight per asset |
| `DAILY_DRAWDOWN_LIMIT` | `0.08` | HaltNewTrades above 8% daily DD |
| `HARD_DRAWDOWN_LIMIT` | `0.15` | CloseAll above 15% drawdown |
| `HARD_LEVERAGE_LIMIT` | `5.0` | CloseAll above this leverage |
| `VOL_SPIKE_THRESHOLD` | `3.0` | ReduceAll if EWMA vol > N× baseline |
| `TARGET_DAILY_VOL` | `0.015` | Vol-targeting denominator |
| `KELLY_FRACTION` | `0.30` | Fractional Kelly scaling |
| `MAX_POSITIONS` | `5` | Maximum concurrent open positions |
| `MAX_SIGNAL_AGE_MS` | `5000` | Reject stale signals older than this |
| `DKF_SIGMA_OBS` | `0.001` | Dual KF observation noise σ |
| `DKF_SIGMA_PARAM_WALK` | `0.00001` | Dual KF parameter random-walk σ |
| `FREQ_WINDOW_BARS` | `64` | Circular buffer size for Goertzel DFT |
| `FUSION_EVIDENCE_DECAY` | `0.99` | EMA decay for log-evidence weights |
| `FUSION_MIN_WEIGHT` | `0.05` | Minimum source weight floor |
| `BACKTEST_TRAIN_BARS` | `500` | Walk-forward training window size |
| `BACKTEST_TEST_BARS` | `100` | Walk-forward test window size |
| `IMPACT_ETA` | `0.1` | Almgren-Chriss permanent impact η |
| `IMPACT_KAPPA` | `0.3` | Almgren-Chriss temporary impact κ |
| `MAX_COMBO_ASSETS` | `3` | Cardinality: maximum legs in one combo order |
| `MAX_COMBO_TC_BPS` | `25.0` | Total TC budget across all legs in bps |
| `ENUMERATE_COMBO_SUBSETS` | `true` | Exhaustive subset search when n_candidates ≤ 10 |
| `HISTORY_DAYS` | `180` | Days of OHLCV to bootstrap at startup |
| `SIGNAL_INTERVAL_SECS` | `300` | Main loop interval (5 minutes) |

---

## Dependencies

| Crate | Purpose |
|---|---|
| `hypersdk` | Hyperliquid REST + WebSocket client |
| `tokio` | Async runtime |
| `nalgebra` | Matrix algebra (Dual Kalman filter) |
| `statrs` | Statistical distributions |
| `parking_lot` | Low-latency `RwLock` for shared state |
| `serde` / `serde_json` | Signal serialisation |
| `uuid` | Unique signal IDs |
| `anyhow` | Error handling |
| `tracing` | Structured logging |
| `dotenvy` | `.env` file loading |
