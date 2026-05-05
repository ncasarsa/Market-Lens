## TODO: Configuration parameters for the Market Lens project

"""
config.py
---------
Single source of truth for all hyperparameters, paths, and constants.
Import this everywhere instead of hardcoding values.

DEV NOTE (small dataset branch):
  Ticker universe is reduced to 15 tickers (3 per sector) for fast iteration.
  Restore the full 50-ticker TICKER_SECTORS dict before merging to main.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR        = Path(__file__).parent
DATA_DIR        = ROOT_DIR / "data"
MODEL_DIR       = ROOT_DIR / "models"
LOG_DIR         = ROOT_DIR / "logs"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

DATASET_PATH    = DATA_DIR / "dataset.parquet"   # cached output of build_dataset()
CHECKPOINT_PATH = MODEL_DIR / "tft_best.ckpt"


# ---------------------------------------------------------------------------
# Ticker universe  (SMALL DATASET — 15 tickers, 3 per sector)
# ---------------------------------------------------------------------------
# Full 50-ticker universe lives on main. This branch uses the 3 highest-
# liquidity / most data-complete names per sector for fast dev iteration.
# Swap back to the full dict before merging.

TICKER_SECTORS = {
    # Tech — mega-cap + semiconductor
    "AAPL": "tech", "MSFT": "tech", "NVDA": "tech",
    # Energy — integrated + services
    "XOM": "energy", "CVX": "energy", "SLB": "energy",
    # Financials — money-center banks + asset mgmt
    "JPM": "financials", "GS": "financials", "BLK": "financials",
    # Defense — top 3 primes
    "LMT": "defense", "RTX": "defense", "NOC": "defense",
    # Utilities — highest liquidity
    "NEE": "utilities", "DUK": "utilities", "SO": "utilities",
}

TICKERS  = list(TICKER_SECTORS.keys())
SECTORS  = list(set(TICKER_SECTORS.values()))


# ---------------------------------------------------------------------------
# Date ranges
# ---------------------------------------------------------------------------

TRAIN_START = "2018-01-01"
TRAIN_END   = "2024-12-31"

# Train / val / test split (by date, not random -- avoids lookahead)
VAL_START   = "2023-01-01"   # last ~2 years = val + test
TEST_START  = "2024-01-01"   # last 1 year = test


# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------

# These are the observed past inputs fed to the TFT
INDICATOR_COLS = [
    # Trend
    "ema_20", "ema_50", "sma_20", "sma_50",
    "macd", "macd_signal", "macd_diff",
    "adx", "cci", "ichimoku_a", "ichimoku_b",
    # Momentum
    "rsi", "stoch_k", "stoch_d", "williams_r",
    "roc", "tsi", "ultimate",
    # Volatility
    "bb_high", "bb_low", "bb_mid", "bb_pct", "bb_width",
    "atr", "kc_high", "kc_low",
    # Volume
    "obv", "vwap", "mfi", "cmf", "force", "eom",
]

OHLCV_COLS = ["open", "high", "low", "close", "volume"]

# All continuous features passed as observed inputs to TFT.
# log_return is intentionally excluded: it is a one-step lag of the target,
# so including it causes direct data leakage into the encoder.
TIME_VARYING_KNOWN_REALS     = ["days_to_fomc", "days_to_earnings", "is_earnings_week"]          # e.g. FOMC dates -- added later
TIME_VARYING_UNKNOWN_REALS   = OHLCV_COLS + INDICATOR_COLS +  [
    "sentiment_score",
    "sentiment_volume",
    "sentiment_rolling_3d",
]

# Static features (don't change over time for a given ticker)
STATIC_CATEGORICALS          = ["ticker", "sector"]
STATIC_REALS                 = []          # e.g. market cap bucket -- future addition

TARGET_COL                   = "target"    # next-day log return
GROUP_COL                    = "ticker"    # PF group identifier
TIME_COL                     = "time_idx"  # PF time identifier


# ---------------------------------------------------------------------------
# TFT hyperparameters (defaults -- overridden by W&B sweep)
# ---------------------------------------------------------------------------

TFT = dict(
    max_encoder_length      = 90,    # lookback window: 60 trading days (~3 months)
    max_prediction_length   = 1,     # predict 1 day ahead
    hidden_size             = 64,
    lstm_layers             = 2,
    dropout                 = 0.2,
    attention_head_size     = 4,
    learning_rate           = 1e-3,
    gradient_clip_val       = 0.1,
)


# ---------------------------------------------------------------------------
# LSTM Autoencoder hyperparameters (regime detection)
# ---------------------------------------------------------------------------

REGIME = dict(
    input_size      = len(OHLCV_COLS) + len(INDICATOR_COLS),
    hidden_size     = 32,
    latent_size     = 8,
    num_layers      = 2,
    dropout         = 0.1,
    n_clusters      = 4,             # trending / mean-reverting / high-vol / low-vol
    sequence_length = 20,            # 20-day windows for regime encoding
)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

TRAIN = dict(
    batch_size          = 128,
    max_epochs          = 50,
    early_stopping_patience = 15,
    num_workers         = 0,         # set >0 on Linux/Colab, keep 0 on Windows
)


# ---------------------------------------------------------------------------
# W&B
# ---------------------------------------------------------------------------

WANDB = dict(
    project        = "marketlens",
    entity         = "bucknell-university-csci357-2026sp",
    user_name      = "Nathan Casarsa",
    user_initials  = "NC",
    user_id        = "njc013",
    user_email     = "njc013@bucknell.edu"
)

SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "hidden_size":          {"values": [32, 64, 128]},
        "lstm_layers":          {"values": [1, 2]},
        "dropout":              {"values": [0.05, 0.1, 0.2]},
        "attention_head_size":  {"values": [2, 4]},
        "learning_rate":        {"min": 1e-4, "max": 1e-2, "distribution": "log_uniform_values"},
        "max_encoder_length":   {"values": [30, 60, 90]},
        "batch_size":           {"values": [64, 128, 256]},
    },
}


# ---------------------------------------------------------------------------
# FinBERT
# ---------------------------------------------------------------------------

FINBERT = dict(
    model_name  = "ProsusAI/finbert",
    max_length  = 512,
    batch_size  = 32,
)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Tickers:          {len(TICKERS)}")
    print(f"Sectors:          {SECTORS}")
    print(f"Indicator cols:   {len(INDICATOR_COLS)}")
    print(f"Unknown reals:    {len(TIME_VARYING_UNKNOWN_REALS)}")
    print(f"Train range:      {TRAIN_START} -> {VAL_START}")
    print(f"Val range:        {VAL_START} -> {TEST_START}")
    print(f"Test range:       {TEST_START} -> {TRAIN_END}")
    print(f"TFT config:       {TFT}")
    print(f"Data dir:         {DATA_DIR}")
    print(f"Model dir:        {MODEL_DIR}")