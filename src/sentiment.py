"""
sentiment.py
------------
Fetches financial news headlines for each ticker via Yahoo Finance RSS,
scores them with FinBERT (ProsusAI/finbert), and returns a daily
sentiment DataFrame that can be joined onto the main price DataFrame.

Three columns are produced per ticker per trading day:
    sentiment_score      : float in [-1, +1]  (positive - negative prob)
    sentiment_volume     : float  number of headlines that day
    sentiment_rolling_3d : float  3-day rolling mean of sentiment_score

These go into TIME_VARYING_UNKNOWN_REALS — they are observed past inputs,
not known future ones (you cannot know tomorrow's headlines today).

Usage:
    from sentiment import build_sentiment, add_sentiment

    # Build and cache (slow — runs FinBERT on all headlines)
    sent_df = build_sentiment(
        tickers   = TICKERS,
        start     = TRAIN_START,
        end       = TRAIN_END,
        save_path = "data/sentiment.parquet",
    )

    # Join onto price DataFrame
    df = add_sentiment(df, sent_df)

Notes:
    - Yahoo Finance RSS gives ~200-500 headlines per ticker going back
      several years. Coverage thins before 2020 for smaller tickers.
    - For dates with zero headlines, sentiment_score is forward-filled
      from the previous trading day (markets carry stale sentiment).
    - FinBERT inference runs on GPU if available, else CPU.
      On A100: ~5-10 min for 15 tickers across 6 years of headlines.
      On CPU: ~30-60 min.
    - Cache the output parquet — do not re-run every session.
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import feedparser
import yfinance as yf

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FINBERT, TICKERS, TRAIN_START, TRAIN_END


# ---------------------------------------------------------------------------
# 1. Headline fetching
# ---------------------------------------------------------------------------

# Yahoo Finance RSS template — returns ~100 recent headlines per ticker
_YF_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"

# Broader market feeds that apply to all tickers (macro sentiment)
_MACRO_RSS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^DJI&region=US&lang=en-US",
]


def _fetch_rss_headlines(ticker: str) -> list[dict]:
    """
    Fetches headlines from Yahoo Finance RSS for a single ticker.
    Returns list of dicts with keys: title, published (datetime).
    Falls back to yfinance .news if RSS returns nothing.
    """
    headlines = []

    # Primary: RSS feed
    try:
        url  = _YF_RSS.format(ticker=ticker)
        feed = feedparser.parse(url)
        for entry in feed.entries:
            try:
                pub = pd.to_datetime(entry.published).normalize().tz_localize(None)
                headlines.append({"title": entry.title, "date": pub})
            except Exception:
                continue
    except Exception:
        pass

    # Fallback: yfinance .news (shorter history but more reliable)
    if len(headlines) < 10:
        try:
            news = yf.Ticker(ticker).news or []
            for item in news:
                try:
                    pub = pd.to_datetime(item["providerPublishTime"], unit="s").normalize().tz_localize(None)
                    headlines.append({"title": item["title"], "date": pub})
                except Exception:
                    continue
        except Exception:
            pass

    return headlines


def fetch_all_headlines(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Fetches headlines for all tickers and returns a long-format DataFrame.

    Columns: ticker, date, title
    """
    start_dt = pd.to_datetime(start).tz_localize("UTC")
    end_dt   = pd.to_datetime(end).tz_localize("UTC")

    frames = []
    for i, ticker in enumerate(tickers):
        print(f"  Fetching headlines [{i+1}/{len(tickers)}]: {ticker}...", end=" ")
        rows = _fetch_rss_headlines(ticker)

        df = pd.DataFrame(rows)
        if df.empty:
            print("no headlines found")
            continue

        df["ticker"] = ticker
        df["date"]   = pd.to_datetime(df["date"]).dt.normalize()
        df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
        df = df.drop_duplicates(subset=["title"])
        frames.append(df)
        print(f"{len(df)} headlines")
        time.sleep(0.3)   # be polite to Yahoo RSS

    if not frames:
        raise RuntimeError("No headlines fetched for any ticker.")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nTotal headlines: {len(combined)} across {combined['ticker'].nunique()} tickers")
    return combined[["ticker", "date", "title"]]


# ---------------------------------------------------------------------------
# 2. FinBERT inference
# ---------------------------------------------------------------------------

class FinBERTScorer:
    """
    Wraps ProsusAI/finbert for batch sentiment scoring.
    Labels: positive (0), negative (1), neutral (2)
    sentiment_score = positive_prob - negative_prob  ∈ [-1, +1]
    """

    def __init__(self, model_name: str = None, batch_size: int = None):
        model_name = model_name or FINBERT["model_name"]
        self.batch_size = batch_size or FINBERT["batch_size"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading FinBERT on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print("FinBERT loaded.")

    @torch.no_grad()
    def score_batch(self, texts: list[str]) -> np.ndarray:
        """
        Scores a list of texts.
        Returns array of shape (N, 3): [positive_prob, negative_prob, neutral_prob]
        """
        all_probs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding     = True,
                truncation  = True,
                max_length  = FINBERT["max_length"],
                return_tensors = "pt",
            ).to(self.device)

            logits = self.model(**enc).logits        # (B, 3)
            probs  = softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

        return np.vstack(all_probs)   # (N, 3)

    def score_dataframe(self, headlines_df: pd.DataFrame) -> pd.DataFrame:
        """
        Scores all headlines in the DataFrame.
        Adds columns: pos_prob, neg_prob, neu_prob, sentiment_score
        """
        print(f"Running FinBERT on {len(headlines_df)} headlines...")
        texts = headlines_df["title"].tolist()
        probs = self.score_batch(texts)

        out = headlines_df.copy()
        out["pos_prob"]        = probs[:, 0]
        out["neg_prob"]        = probs[:, 1]
        out["neu_prob"]        = probs[:, 2]
        out["sentiment_score"] = out["pos_prob"] - out["neg_prob"]
        return out


# ---------------------------------------------------------------------------
# 3. Aggregate to daily scores
# ---------------------------------------------------------------------------

def aggregate_daily_sentiment(
    scored_df: pd.DataFrame,
    all_trading_days: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Aggregates per-headline scores to one row per (ticker, trading_day).

    For each ticker × day:
        sentiment_score  = mean(positive_prob - negative_prob)
        sentiment_volume = number of headlines

    Days with no headlines are filled forward (stale sentiment) then
    back-filled at the start of each ticker's series.

    Returns DataFrame with columns:
        ticker, date, sentiment_score, sentiment_volume, sentiment_rolling_3d
    """
    daily = (
        scored_df
        .groupby(["ticker", "date"])
        .agg(
            sentiment_score  = ("sentiment_score", "mean"),
            sentiment_volume = ("sentiment_score", "count"),
        )
        .reset_index()
    )

    # Build full (ticker × trading_day) grid so every date has a row
    tickers = daily["ticker"].unique()
    grid = pd.MultiIndex.from_product(
        [tickers, all_trading_days],
        names=["ticker", "date"],
    )
    daily = (
        daily
        .set_index(["ticker", "date"])
        .reindex(grid)
        .reset_index()
    )

    # Fill volume NaN → 0 (no headlines that day)
    daily["sentiment_volume"] = daily["sentiment_volume"].fillna(0.0)

    # Forward-fill score per ticker (carry stale sentiment), then backfill
    daily = daily.sort_values(["ticker", "date"])
    daily["sentiment_score"] = (
        daily.groupby("ticker")["sentiment_score"]
        .transform(lambda x: x.ffill().bfill())
    )

    # Neutral fallback for tickers with no headlines at all
    daily["sentiment_score"] = daily["sentiment_score"].fillna(0.0)

    # 3-day rolling mean (smooths single-day spikes)
    daily["sentiment_rolling_3d"] = (
        daily.groupby("ticker")["sentiment_score"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    return daily[["ticker", "date", "sentiment_score",
                  "sentiment_volume", "sentiment_rolling_3d"]]


# ---------------------------------------------------------------------------
# 4. Master pipeline
# ---------------------------------------------------------------------------

def build_sentiment(
    tickers: list[str] = None,
    start: str = TRAIN_START,
    end: str   = TRAIN_END,
    save_path: str = None,
) -> pd.DataFrame:
    """
    Full pipeline: fetch headlines → FinBERT scoring → daily aggregation.

    Args:
        tickers:   ticker list (defaults to config.TICKERS)
        start:     start date string "YYYY-MM-DD"
        end:       end date string "YYYY-MM-DD"
        save_path: if provided, caches result as parquet

    Returns:
        DataFrame with columns:
            ticker, date, sentiment_score, sentiment_volume, sentiment_rolling_3d
    """
    tickers = tickers or TICKERS
    print(f"\nBuilding sentiment dataset for {len(tickers)} tickers "
          f"({start} → {end})...")

    # Step 1: fetch headlines
    headlines_df = fetch_all_headlines(tickers, start, end)

    # Step 2: FinBERT scoring
    scorer    = FinBERTScorer()
    scored_df = scorer.score_dataframe(headlines_df)

    # Step 3: aggregate to daily
    all_trading_days = pd.bdate_range(start=start, end=end)
    daily_df = aggregate_daily_sentiment(scored_df, all_trading_days)

    if save_path:
        daily_df.to_parquet(save_path, index=False)
        print(f"  Saved sentiment to {save_path}")

    print(f"\nSentiment dataset: {len(daily_df)} rows, "
          f"{daily_df['ticker'].nunique()} tickers")
    return daily_df


# ---------------------------------------------------------------------------
# 5. Join onto main price DataFrame
# ---------------------------------------------------------------------------

def add_sentiment(
    df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-joins daily sentiment scores onto the main price DataFrame.

    Rows in df with no sentiment match (e.g. dates before RSS history)
    are filled with 0.0 (neutral) for score/rolling and 0 for volume.

    Args:
        df:            cleaned price DataFrame from data.py
        sentiment_df:  output of build_sentiment()

    Returns:
        df with three new columns:
            sentiment_score, sentiment_volume, sentiment_rolling_3d
    """
    sentiment_df = sentiment_df.copy()
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])

    merged = df.merge(
        sentiment_df[["ticker", "date",
                      "sentiment_score", "sentiment_volume",
                      "sentiment_rolling_3d"]],
        on  = ["ticker", "date"],
        how = "left",
    )

    # Fill any unmatched rows with neutral values
    merged["sentiment_score"]      = merged["sentiment_score"].fillna(0.0)
    merged["sentiment_volume"]     = merged["sentiment_volume"].fillna(0.0)
    merged["sentiment_rolling_3d"] = merged["sentiment_rolling_3d"].fillna(0.0)

    print(f"  Sentiment joined. Coverage: "
          f"{(merged['sentiment_volume'] > 0).mean():.1%} of rows have headlines.")
    return merged


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    tickers = ["AAPL", "MSFT", "NVDA"]

    sent = build_sentiment(
        tickers   = tickers,
        start     = "2023-01-01",
        end       = "2023-12-31",
        save_path = None,
    )
    print(sent.head(10))
    print("\nScore distribution:")
    print(sent["sentiment_score"].describe())