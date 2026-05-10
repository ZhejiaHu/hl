/// The traded asset universe with their Hyperliquid asset indices and metadata.
#[derive(Debug, Clone)]
pub struct Asset {
    pub symbol: String,
    pub index: usize,
    pub max_leverage: u32,
    pub sz_decimals: u32,
    pub max_weight: f64,
}

impl Asset {
    fn new(symbol: &str, index: usize, max_leverage: u32, sz_decimals: u32, max_weight: f64) -> Self {
        Self {
            symbol: symbol.to_string(),
            index,
            max_leverage,
            sz_decimals,
            max_weight,
        }
    }
}

/// The full asset universe traded by this system.
///
/// BTC and ETH are the liquidity anchors and regime benchmarks.
/// SOL and HYPE provide ecosystem-specific momentum alpha.
/// ARB and MATIC diversify into L2/DeFi beta.
/// WIF is a meme-momentum signal asset with a hard weight cap.
pub fn universe() -> Vec<Asset> {
    vec![
        Asset::new("BTC",   0,   50, 5, 0.30),
        Asset::new("ETH",   1,   50, 4, 0.25),
        Asset::new("SOL",   2,   20, 2, 0.15),
        Asset::new("HYPE", 107,  10, 2, 0.10),
        Asset::new("ARB",   8,   10, 0, 0.10),
        Asset::new("MATIC", 6,   10, 0, 0.10),
        Asset::new("WIF",  30,   10, 0, 0.05),
    ]
}

/// Assets used as macro anchors but not directly traded.
pub fn macro_anchors() -> Vec<&'static str> {
    vec!["BTC", "ETH"]
}
