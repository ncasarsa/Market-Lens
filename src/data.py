"""
data.py
-------
Ingests OHLCV data via yfinance, computes technical indicators via `ta`,
and returns a clean long-format DataFrame ready for PyTorch Forecasting's
TimeSeriesDataSet.

Output schema (one row per ticker per trading day):
    time_idx        : int  — monotonic integer time index (required by PF)
    ticker          : str  — stock symbol (group_id in PF)
    sector          : str  — static categorical covariate
    date            : date — human-readable date (not used by PF directly)
    open/high/low/close/volume : float — raw OHLCV
    [30+ indicator columns]   : float — technical features
    target          : float — next-day log return (what we predict)
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import feedparser
from ta.utils import dropna

# ---------------------------------------------------------------------------
# Ticker universe — mirrors config.py (imported here for standalone use too)
# ---------------------------------------------------------------------------
TICKER_SECTORS = {
    # Tech
    "AAPL": "tech", "MSFT": "tech", "NVDA": "tech", "GOOGL": "tech",
    "META": "tech", "AMD": "tech", "INTC": "tech", "AVGO": "tech",
    "ORCL": "tech", "CRM": "tech",
    # Energy
    "XOM": "energy", "CVX": "energy", "COP": "energy", "SLB": "energy",
    "EOG": "energy", "MPC": "energy", "PSX": "energy", "VLO": "energy",
    "OXY": "energy", "HAL": "energy",
    # Financials
    "JPM": "financials", "BAC": "financials", "WFC": "financials",
    "GS": "financials", "MS": "financials", "C": "financials",
    "BLK": "financials", "SCHW": "financials", "AXP": "financials",
    "COF": "financials",
    # Defense
    "LMT": "defense", "RTX": "defense", "NOC": "defense", "GD": "defense",
    "BA": "defense", "HII": "defense", "L3H": "defense", "TDG": "defense",
    "HEI": "defense", "LDOS": "defense",
    # Utilities
    "NEE": "utilities", "DUK": "utilities", "SO": "utilities",
    "D": "utilities", "AEP": "utilities", "EXC": "utilities",
    "SRE": "utilities", "PCG": "utilities", "ES": "utilities",
    "ETR": "utilities",
}

DEFAULT_START = "2018-01-01"
DEFAULT_END   = "2024-12-31"


# ---------------------------------------------------------------------------
# 1. Raw OHLCV download
# ---------------------------------------------------------------------------

def download_ohlcv(
    tickers: list[str] = None,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> pd.DataFrame:
    """
    Downloads daily OHLCV for all tickers and returns a long-format DataFrame.
    Drops tickers that fail to download or have >5% missing close values.
    """
    if tickers is None:
        tickers = list(TICKER_SECTORS.keys())

    print(f"Downloading {len(tickers)} tickers from {start} to {end}...")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,   # adjusts for splits/dividends
        progress=False,
    )

    # yfinance returns MultiIndex columns (Price, Ticker) when >1 ticker
    # Reshape to long format
    frames = []
    for ticker in tickers:
        try:
            df = raw.xs(ticker, axis=1, level=1).copy()
        except KeyError:
            print(f"  WARNING: {ticker} not found, skipping.")
            continue

        df.columns = [c.lower() for c in df.columns]
        missing = df["close"].isna().mean()
        if missing > 0.05:
            print(f"  WARNING: {ticker} has {missing:.1%} missing close values, skipping.")
            continue

        df = df.dropna(subset=["close"])
        df["ticker"] = ticker
        df["sector"] = TICKER_SECTORS.get(ticker, "unknown")
        df.index.name = "date"
        df = df.reset_index()
        frames.append(df)

    if not frames:
        raise RuntimeError("No tickers downloaded successfully.")

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"  Downloaded {combined['ticker'].nunique()} tickers, "
          f"{len(combined)} rows total.")
    return combined


def add_vix(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    Downloads VIX and joins vix_normalized and vix_change onto the main
    price DataFrame. Both go into TIME_VARYING_KNOWN_REALS — VIX futures
    prices are publicly known in advance.
    """
    print("Downloading VIX...")
    try:
        vix = yf.download("^VIX", start=start, end=end,
                          auto_adjust=True, progress=False)
        vix = vix["Close"].reset_index()
        vix.columns = ["date", "vix"]
        vix["date"]           = pd.to_datetime(vix["date"])
        vix["vix_normalized"] = (vix["vix"] - vix["vix"].mean()) / vix["vix"].std()
        vix["vix_change"]     = vix["vix"].pct_change().fillna(0.0)

        df = df.merge(vix[["date", "vix_normalized", "vix_change"]],
                      on="date", how="left")
        df["vix_normalized"] = df["vix_normalized"].ffill().bfill()
        df["vix_change"]     = df["vix_change"].ffill().bfill()
        print(f"  VIX joined. Coverage: {df['vix_normalized'].notna().mean():.1%}")
    except Exception as e:
        print(f"  VIX download failed ({e}) — filling with zeros")
        df["vix_normalized"] = 0.0
        df["vix_change"]     = 0.0
    return df

def add_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    print("log_return present:", "log_return" in df.columns)
    print("columns:", [c for c in df.columns if 'return' in c.lower() or 'log' in c.lower()])
    """
    Adds cross-sectional relative strength features per trading day.
    These capture how each ticker is performing relative to its sector
    and the broader market — documented return predictors that the TFT
    cannot derive from single-ticker history alone.

    Columns added:
        sector_rel_return  : return minus sector mean that day
        market_rel_return  : return minus market mean that day
        sector_rel_20d     : 20-day rolling mean of sector_rel_return
        market_rel_20d     : 20-day rolling mean of market_rel_return
    """
    print("Adding cross-sectional features...")
    df = df.copy()

    # Daily relative returns — computed cross-sectionally per date
    df["sector_rel_return"] = df.groupby(["date", "sector"])["log_return"].transform(
        lambda x: x - x.mean()
    )
    df["market_rel_return"] = df.groupby("date")["log_return"].transform(
        lambda x: x - x.mean()
    )

    # Rolling 20-day mean — medium-term momentum signal
    df["sector_rel_20d"] = df.groupby("ticker")["sector_rel_return"].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    df["market_rel_20d"] = df.groupby("ticker")["market_rel_return"].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )

    # Fill any NaNs from rolling window start
    for col in ["sector_rel_return", "market_rel_return",
                "sector_rel_20d", "market_rel_20d"]:
        df[col] = df.groupby("ticker")[col].transform(
            lambda x: x.ffill().fillna(0.0)
        )

    print(f"  Cross-sectional features added.")
    return df

# ---------------------------------------------------------------------------
# 2. Technical indicators
# ---------------------------------------------------------------------------

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes 30+ technical indicators per ticker using the `ta` library.
    All indicators are added as new columns in-place (per ticker group).
    """
    print("Computing technical indicators...")
    result_frames = []

    for ticker, group in df.groupby("ticker"):
        g = group.copy().sort_values("date")

        # Ensure no NaNs in OHLCV before computing indicators
        g = dropna(g)

        close = g["close"]
        high  = g["high"]
        low   = g["low"]
        volume = g["volume"]

        # --- Trend ---
        g["ema_20"]     = ta.trend.ema_indicator(close, window=20)
        g["ema_50"]     = ta.trend.ema_indicator(close, window=50)
        g["sma_20"]     = ta.trend.sma_indicator(close, window=20)
        g["sma_50"]     = ta.trend.sma_indicator(close, window=50)
        macd            = ta.trend.MACD(close)
        g["macd"]       = macd.macd()
        g["macd_signal"]= macd.macd_signal()
        g["macd_diff"]  = macd.macd_diff()
        g["adx"]        = ta.trend.adx(high, low, close, window=14)
        g["cci"]        = ta.trend.cci(high, low, close, window=20)
        ich             = ta.trend.IchimokuIndicator(high, low)
        g["ichimoku_a"] = ich.ichimoku_a()
        g["ichimoku_b"] = ich.ichimoku_b()

        # --- Momentum ---
        g["rsi"]        = ta.momentum.rsi(close, window=14)
        stoch           = ta.momentum.StochasticOscillator(high, low, close)
        g["stoch_k"]    = stoch.stoch()
        g["stoch_d"]    = stoch.stoch_signal()
        g["williams_r"] = ta.momentum.williams_r(high, low, close, lbp=14)
        g["roc"]        = ta.momentum.roc(close, window=12)
        g["tsi"]        = ta.momentum.tsi(close)
        g["ultimate"]   = ta.momentum.ultimate_oscillator(high, low, close)

        # --- Volatility ---
        bb              = ta.volatility.BollingerBands(close, window=20)
        g["bb_high"]    = bb.bollinger_hband()
        g["bb_low"]     = bb.bollinger_lband()
        g["bb_mid"]     = bb.bollinger_mavg()
        g["bb_pct"]     = bb.bollinger_pband()   # %B: position within bands
        g["bb_width"]   = bb.bollinger_wband()
        g["atr"]        = ta.volatility.average_true_range(high, low, close, window=14)
        g["kc_high"]    = ta.volatility.keltner_channel_hband(high, low, close)
        g["kc_low"]     = ta.volatility.keltner_channel_lband(high, low, close)

        # --- Volume ---
        g["obv"]        = ta.volume.on_balance_volume(close, volume)
        g["vwap"]       = ta.volume.volume_weighted_average_price(high, low, close, volume)
        g["mfi"]        = ta.volume.money_flow_index(high, low, close, volume, window=14)
        g["cmf"]        = ta.volume.chaikin_money_flow(high, low, close, volume, window=20)
        g["force"]      = ta.volume.force_index(close, volume)
        g["eom"]        = ta.volume.ease_of_movement(high, low, volume, window=14)

        result_frames.append(g)

    combined = pd.concat(result_frames, ignore_index=True)
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"  Added indicators. Shape: {combined.shape}")
    return combined


# ---------------------------------------------------------------------------
# 3. Target: next-day log return
# ---------------------------------------------------------------------------

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds `target` column: next-day log return, computed per ticker.
    The last row per ticker will have NaN target and should be dropped
    before training (but kept for live inference).
    """
    df = df.copy()
    df["log_return"] = df.groupby("ticker")["close"].transform(
        lambda x: np.log(x).diff()
    )
    for lag in [1, 2, 3, 5, 10]:
        df[f"return_lag_{lag}"] = df.groupby("ticker")["log_return"].shift(lag)

    # Shift log_return back by 1 so each row's target = tomorrow's return
    df["target"] = df.groupby("ticker")["log_return"].shift(-1)
    return df


# ---------------------------------------------------------------------------
# 4. Time index (required by PyTorch Forecasting)
# ---------------------------------------------------------------------------

def add_time_idx(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds `time_idx`: a monotonic integer index over the full date range
    shared across all tickers (not per-ticker). PF requires this to be
    the same across groups so temporal alignment is consistent.
    """
    df = df.copy()
    all_dates = sorted(df["date"].unique())
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    df["time_idx"] = df["date"].map(date_to_idx)
    return df


# ---------------------------------------------------------------------------
# 4b. Known future covariates: FOMC dates + earnings calendar
# ---------------------------------------------------------------------------

# FOMC meeting dates 2018-2024 (actual decision days, sourced from federalreserve.gov)
FOMC_DATES = pd.Series(pd.to_datetime([
    "2018-01-31","2018-03-21","2018-05-02","2018-06-13",
    "2018-08-01","2018-09-26","2018-11-08","2018-12-19",
    "2019-01-30","2019-03-20","2019-05-01","2019-06-19",
    "2019-07-31","2019-09-18","2019-10-30","2019-12-11",
    "2020-01-29","2020-03-03","2020-03-15","2020-04-29",
    "2020-06-10","2020-07-29","2020-09-16","2020-11-05",
    "2020-12-16","2021-01-27","2021-03-17","2021-04-28",
    "2021-06-16","2021-07-28","2021-09-22","2021-11-03",
    "2021-12-15","2022-01-26","2022-03-16","2022-05-04",
    "2022-06-15","2022-07-27","2022-09-21","2022-11-02",
    "2022-12-14","2023-02-01","2023-03-22","2023-05-03",
    "2023-06-14","2023-07-26","2023-09-20","2023-11-01",
    "2023-12-13","2024-01-31","2024-03-20","2024-05-01",
    "2024-06-12","2024-07-31","2024-09-18","2024-11-07",
    "2024-12-18",
]))

def add_known_future_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds time-varying known future covariates to the DataFrame:
      - days_to_fomc     : trading days until next FOMC decision (capped at 30)
      - days_to_earnings : trading days until next earnings release for that ticker (capped at 30)
      - is_earnings_week : 1 if within 2 trading days of an earnings release, else 0

    These go into TIME_VARYING_KNOWN_REALS because the dates are publicly
    known in advance — the TFT can see them in the decoder window.

    Earnings dates are fetched from yfinance. If a ticker fails, its
    days_to_earnings and is_earnings_week columns default to 30 and 0.
    """
    print("Adding known future covariates (FOMC + earnings)...")
    df = df.copy()
    all_dates = sorted(df["date"].unique())
    all_dates_series = pd.Series(all_dates)

    # ── FOMC countdown (same for all tickers, computed once) ────────────────
    def days_to_next_fomc(date):
        future = FOMC_DATES[FOMC_DATES >= date]
        if len(future) == 0:
            return 30  # past last known date — cap at 30
        # Count trading days, not calendar days
        trading_days = all_dates_series[all_dates_series >= date]
        next_fomc = future.iloc[0]
        n = (trading_days <= next_fomc).sum() - 1  # -1 so meeting day = 0
        return int(min(n, 30))

    date_to_fomc = {d: days_to_next_fomc(d) for d in all_dates}
    df["days_to_fomc"] = df["date"].map(date_to_fomc).astype(float)

    # ── Earnings countdown (per ticker) ─────────────────────────────────────
    earnings_lookup: dict[str, pd.DatetimeIndex] = {}

    for ticker in df["ticker"].unique():
        try:
            cal = yf.Ticker(ticker).get_earnings_dates(limit=40)
            if cal is None or cal.empty:
                raise ValueError("empty")
            edates = pd.to_datetime(cal.index).normalize().sort_values()
            # Keep only dates within our training window
            edates = edates[
                (edates >= df["date"].min()) & (edates <= df["date"].max())
            ]
            earnings_lookup[ticker] = edates
        except Exception:
            earnings_lookup[ticker] = pd.DatetimeIndex([])

    rows_by_ticker = df.groupby("ticker")

    result_frames = []
    for ticker, group in rows_by_ticker:
        group = group.copy().sort_values("date")
        edates = earnings_lookup.get(ticker, pd.DatetimeIndex([]))
        ticker_dates = group["date"].values

        dtе_arr = []
        iew_arr = []

        for date in ticker_dates:
            if len(edates) == 0:
                dtе_arr.append(30.0)
                iew_arr.append(0.0)
                continue

            future_e = edates[edates >= date]
            if len(future_e) == 0:
                dtе_arr.append(30.0)
            else:
                next_e = future_e[0]
                td = all_dates_series[all_dates_series >= date]
                n = int((td <= next_e).sum()) - 1
                dtе_arr.append(float(min(n, 30)))

            # is_earnings_week: within 2 trading days in either direction
            near = edates[
                (edates >= date - pd.Timedelta(days=4)) &
                (edates <= date + pd.Timedelta(days=4))
            ]
            iew_arr.append(1.0 if len(near) > 0 else 0.0)

        group["days_to_earnings"] = dtе_arr
        group["is_earnings_week"] = iew_arr
        result_frames.append(group)

    df = pd.concat(result_frames).sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"  FOMC range: {df['days_to_fomc'].min():.0f}–{df['days_to_fomc'].max():.0f} days")
    print(f"  Earnings week rows: {df['is_earnings_week'].sum():.0f} / {len(df)}")
    return df

# ---------------------------------------------------------------------------
# 5. Normalize / clean
# ---------------------------------------------------------------------------

def clean_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Drops rows with NaN target (last row per ticker)
    - Forward-fills remaining NaNs in indicator columns (edge of window)
    - Clips extreme outliers in log_return (>5 std)
    """
    df = df.copy()

    # Drop rows with no target (can't train on them)
    df = df.dropna(subset=["target"])

    # Forward fill indicator NaNs (from warm-up windows like EMA-50)
    indicator_cols = [c for c in df.columns if c not in
                  ["date", "ticker", "sector", "time_idx",
                   "openx`", "high", "low", "close", "volume",
                   "log_return", "target",
                   "days_to_fomc", "days_to_earnings", "is_earnings_week",
                   "sentiment_score", "sentiment_volume", "sentiment_rolling_3d",
                   "return_lag_1", "return_lag_2", "return_lag_3",
                   "return_lag_5", "return_lag_10", "sector_rel_return", "market_rel_return",
                   "sector_rel_20d", "market_rel_20d"]]
    df[indicator_cols] = df.groupby("ticker")[indicator_cols].transform(
        lambda x: x.ffill()
    )

    # Drop any remaining NaN rows (start of series before any ffill can help)
    df = df.dropna(subset=indicator_cols)

    df["target"] = df["target"] / 0.02   # scale by typical daily vol
    
    # Clip extreme z-scores after normalization
    df["target"] = df["target"].clip(-5, 5)

    # Replace any inf/-inf with NaN then forward fill (catches OBV blow-ups etc.)
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].replace([float("inf"), float("-inf")], float("nan"))
    df[numeric_cols] = df.groupby("ticker")[numeric_cols].transform(lambda x: x.ffill().bfill())

    # Final drop of any remaining nulls
    df = df.dropna()

    df = df.sort_values(["ticker", "time_idx"]).reset_index(drop=True)
    print(f"  After cleaning: {len(df)} rows, {df['ticker'].nunique()} tickers.")
    return df


# ---------------------------------------------------------------------------
# 6. Master pipeline
# ---------------------------------------------------------------------------

def build_dataset(
    tickers: list[str] = None,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    save_path: str = None,
    sentiment_path: str = None,
) -> pd.DataFrame:
    """
    Full pipeline: download → indicators → target → time_idx → clean.

    Returns a long-format DataFrame ready to pass into TimeSeriesDataSet.

    Args:
        tickers:   list of ticker symbols (defaults to full TICKER_SECTORS universe)
        start:     start date string "YYYY-MM-DD"
        end:       end date string "YYYY-MM-DD"
        save_path: if provided, saves the result to this path as parquet

    Returns:
        pd.DataFrame with columns:
            time_idx, ticker, sector, date, open, high, low, close, volume,
            [indicators...], log_return, target
    """
    df = download_ohlcv(tickers, start, end)
    df = add_vix(df, start, end)
    df = add_technical_indicators(df)
    df = add_target(df)
    df = add_time_idx(df)
    df = add_cross_sectional_features(df)
    df = add_known_future_covariates(df)
    if sentiment_path:
        from pathlib import Path
        from sentiment import build_sentiment, add_sentiment
        sent_path = Path(sentiment_path)
        if sent_path.exists():
            print(f"Loading cached sentiment from {sent_path}...")
            sent_df = pd.read_parquet(sent_path)
        else:
            sent_df = build_sentiment(
                tickers   = tickers or list(TICKER_SECTORS.keys()),
                start     = start,
                end       = end,
                save_path = str(sent_path),
            )
        df = add_sentiment(df, sent_df)
    df = clean_and_normalize(df)

    if save_path:
        df.to_parquet(save_path, index=False)
        print(f"  Saved to {save_path}")

    print(f"\nFinal dataset: {df.shape[0]} rows x {df.shape[1]} cols, "
          f"{df['ticker'].nunique()} tickers, "
          f"{df['date'].min().date()} to {df['date'].max().date()}")
    return df


# ---------------------------------------------------------------------------
# Mock data generator (for offline / CI testing without yfinance access)
# ---------------------------------------------------------------------------

def _make_mock_ohlcv(
    tickers: list[str] = None,
    start: str = "2020-01-01",
    end: str = "2023-12-31",
) -> pd.DataFrame:
    """
    Generates synthetic OHLCV data that mirrors the schema of download_ohlcv().
    Used for testing the indicator and pipeline logic without network access.
    """
    if tickers is None:
        tickers = ["AAPL", "MSFT", "XOM", "JPM", "NEE"]

    rng = pd.date_range(start=start, end=end, freq="B")  # business days
    frames = []
    np.random.seed(42)

    for ticker in tickers:
        n = len(rng)
        # Geometric Brownian Motion for close price
        returns = np.random.normal(0.0005, 0.015, n)
        close = 100 * np.exp(np.cumsum(returns))
        high  = close * (1 + np.abs(np.random.normal(0, 0.008, n)))
        low   = close * (1 - np.abs(np.random.normal(0, 0.008, n)))
        open_ = close * (1 + np.random.normal(0, 0.005, n))
        volume = np.random.randint(1_000_000, 50_000_000, n).astype(float)

        frames.append(pd.DataFrame({
            "date":   rng,
            "open":   open_,
            "high":   high,
            "low":    low,
            "close":  close,
            "volume": volume,
            "ticker": ticker,
            "sector": TICKER_SECTORS.get(ticker, "tech"),
        }))

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"  [MOCK] Generated {len(combined)} rows for {len(tickers)} tickers.")
    return combined


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    test_tickers = ["AAPL", "MSFT", "XOM", "JPM", "NEE"]
    mock = "--mock" in sys.argv

    if mock:
        print("Running with mock data (no network required)...")
        raw = _make_mock_ohlcv(test_tickers, "2020-01-01", "2023-12-31")
        df = add_technical_indicators(raw)
        df = add_target(df)
        df = add_time_idx(df)
        df = clean_and_normalize(df)
    else:
        df = build_dataset(tickers=test_tickers, start="2020-01-01", end="2023-12-31")

    print("\nSample output:")
    print(df[["date", "ticker", "sector", "time_idx", "close", "rsi", "macd", "target"]].head(10))
    print("\nColumn list:")
    print(sorted(df.columns.tolist()))
    print("\nDtypes:")
    print(df.dtypes)
    print("\nNull counts:")
    print(df.isnull().sum()[df.isnull().sum() > 0])