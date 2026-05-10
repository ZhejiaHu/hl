//! Technical indicator computations on OHLCV bar series.
//!
//! All functions operate on raw `f64` slices (closes, highs, lows, volumes)
//! and return the most-recent value or `None` if insufficient data.

/// Relative Strength Index (Wilder smoothing).
pub fn rsi(closes: &[f64], period: usize) -> Option<f64> {
    if closes.len() < period + 1 {
        return None;
    }
    let tail = &closes[closes.len() - period - 1..];
    let mut gains = 0.0f64;
    let mut losses = 0.0f64;
    for i in 1..tail.len() {
        let diff = tail[i] - tail[i - 1];
        if diff > 0.0 {
            gains += diff;
        } else {
            losses -= diff;
        }
    }
    let avg_gain = gains / period as f64;
    let avg_loss = losses / period as f64;
    if avg_loss < 1e-15 {
        return Some(100.0);
    }
    let rs = avg_gain / avg_loss;
    Some(100.0 - 100.0 / (1.0 + rs))
}

/// Exponential moving average (single pass).
pub fn ema(series: &[f64], period: usize) -> Option<f64> {
    if series.is_empty() || period == 0 {
        return None;
    }
    let k = 2.0 / (period as f64 + 1.0);
    let mut val = series[0];
    for &x in &series[1..] {
        val = x * k + val * (1.0 - k);
    }
    Some(val)
}

/// MACD line, signal line, and histogram using the most recent values.
/// Returns `(macd, signal, histogram)`.
pub fn macd(closes: &[f64]) -> Option<(f64, f64, f64)> {
    if closes.len() < 35 {
        return None;
    }
    let fast = ema(closes, 12)?;
    let slow = ema(closes, 26)?;
    let macd_line = fast - slow;

    // Signal is a 9-period EMA of the MACD line — approximate with a rolling series.
    let n = closes.len();
    let macd_series: Vec<f64> = (9..=n)
        .filter_map(|i| {
            let slice = &closes[..i];
            let f = ema(slice, 12)?;
            let s = ema(slice, 26)?;
            Some(f - s)
        })
        .collect();
    let signal = ema(&macd_series, 9)?;
    let histogram = macd_line - signal;
    Some((macd_line, signal, histogram))
}

/// Bollinger Band position: (close - upper_band) / (upper_band - lower_band).
/// Returns value in [-1, 1] roughly, 0.5 means at mid-band.
pub fn bollinger_position(closes: &[f64], period: usize, std_mult: f64) -> Option<f64> {
    if closes.len() < period {
        return None;
    }
    let tail = &closes[closes.len() - period..];
    let mu = tail.iter().sum::<f64>() / period as f64;
    let sigma = (tail.iter().map(|&x| (x - mu).powi(2)).sum::<f64>() / period as f64).sqrt();
    if sigma < 1e-15 {
        return Some(0.0);
    }
    let last = *closes.last()?;
    let upper = mu + std_mult * sigma;
    let lower = mu - std_mult * sigma;
    let band_width = upper - lower;
    if band_width < 1e-15 {
        return Some(0.0);
    }
    Some((last - lower) / band_width)
}

/// Average True Range (Wilder).
pub fn atr(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Option<f64> {
    let n = closes.len();
    if n < period + 1 || highs.len() < n || lows.len() < n {
        return None;
    }
    let tail_h = &highs[n - period - 1..];
    let tail_l = &lows[n - period - 1..];
    let tail_c = &closes[n - period - 1..];

    let trs: Vec<f64> = (1..=period)
        .map(|i| {
            let tr = (tail_h[i] - tail_l[i])
                .max((tail_h[i] - tail_c[i - 1]).abs())
                .max((tail_l[i] - tail_c[i - 1]).abs());
            tr
        })
        .collect();

    Some(trs.iter().sum::<f64>() / period as f64)
}

/// Rate of change: (close_t / close_{t-n}) - 1.
pub fn roc(closes: &[f64], period: usize) -> Option<f64> {
    let n = closes.len();
    if n <= period {
        return None;
    }
    let base = closes[n - 1 - period];
    if base.abs() < 1e-15 {
        return None;
    }
    Some(closes[n - 1] / base - 1.0)
}

/// Log volume z-score over the last `lookback` bars.
pub fn volume_z_score(volumes: &[f64], lookback: usize) -> Option<f64> {
    let n = volumes.len();
    if n < lookback {
        return None;
    }
    let tail: Vec<f64> = volumes[n - lookback..]
        .iter()
        .map(|&v| (v + 1.0).ln())
        .collect();
    let mu = tail.iter().sum::<f64>() / tail.len() as f64;
    let sigma = (tail.iter().map(|&x| (x - mu).powi(2)).sum::<f64>() / tail.len() as f64)
        .sqrt()
        .max(1e-10);
    Some((tail.last()? - mu) / sigma)
}

/// Distance from rolling max: (max - close) / max.  Higher = further below high.
pub fn dist_from_rolling_high(closes: &[f64], period: usize) -> Option<f64> {
    let n = closes.len();
    if n < period {
        return None;
    }
    let rolling_max = closes[n - period..]
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let last = *closes.last()?;
    if rolling_max < 1e-15 {
        return Some(0.0);
    }
    Some((rolling_max - last) / rolling_max)
}

/// Distance from rolling min: (close - min) / min.  Higher = further above low.
pub fn dist_from_rolling_low(closes: &[f64], period: usize) -> Option<f64> {
    let n = closes.len();
    if n < period {
        return None;
    }
    let rolling_min = closes[n - period..]
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let last = *closes.last()?;
    if rolling_min.abs() < 1e-15 {
        return Some(0.0);
    }
    Some((last - rolling_min) / rolling_min)
}

/// Pearson correlation between two return series over `period` bars.
pub fn rolling_correlation(r1: &[f64], r2: &[f64], period: usize) -> Option<f64> {
    let n1 = r1.len();
    let n2 = r2.len();
    let n = n1.min(n2);
    if n < period {
        return None;
    }
    let tail1 = &r1[n - period..];
    let tail2 = &r2[n - period..];
    let mu1 = tail1.iter().sum::<f64>() / period as f64;
    let mu2 = tail2.iter().sum::<f64>() / period as f64;
    let cov: f64 = tail1.iter().zip(tail2).map(|(&x, &y)| (x - mu1) * (y - mu2)).sum();
    let var1: f64 = tail1.iter().map(|&x| (x - mu1).powi(2)).sum();
    let var2: f64 = tail2.iter().map(|&x| (x - mu2).powi(2)).sum();
    let denom = (var1 * var2).sqrt();
    if denom < 1e-15 {
        return Some(0.0);
    }
    Some(cov / denom)
}

/// Rolling beta of `asset_returns` relative to `benchmark_returns`.
pub fn rolling_beta(
    asset_returns: &[f64],
    benchmark_returns: &[f64],
    period: usize,
) -> Option<f64> {
    let n = asset_returns.len().min(benchmark_returns.len());
    if n < period {
        return None;
    }
    let ra = &asset_returns[n - period..];
    let rb = &benchmark_returns[n - period..];
    let mu_b = rb.iter().sum::<f64>() / period as f64;
    let mu_a = ra.iter().sum::<f64>() / period as f64;
    let cov: f64 = ra.iter().zip(rb).map(|(&a, &b)| (a - mu_a) * (b - mu_b)).sum();
    let var_b: f64 = rb.iter().map(|&b| (b - mu_b).powi(2)).sum();
    if var_b < 1e-15 {
        return Some(0.0);
    }
    Some(cov / var_b)
}
