"""
regime.py
---------
Post-training pipeline for market regime detection.

After train_regime_ae() produces a trained RegimeAutoencoder, this module:
    1. Encodes all windows into latent vectors
    2. Fits KMeans on those vectors
    3. Labels each cluster semantically (bull / bear / volatile / sideways)
       using rank-based assignment on intra-window mean log return and std
    4. Merges a "regime" string column back onto the main price DataFrame

Usage:
    from regime import fit_regimes, attach_regime_labels

    ae_model, windows, metadata = train_regime_ae(df)
    cluster_map = fit_regimes(ae_model, windows, metadata, df)
    df = attach_regime_labels(df, cluster_map)

Regime label semantics:
    volatile   highest intra-window return std (most chaotic)
    bull       highest intra-window mean return among remaining 3
    bear       lowest  intra-window mean return among remaining 3
    sideways   the one left over
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config import REGIME, DATA_DIR, OHLCV_COLS, INDICATOR_COLS, TARGET_COL


# ---------------------------------------------------------------------------
# 1. Encode all windows into latent space
# ---------------------------------------------------------------------------

def extract_latents(
    model,
    windows: torch.Tensor,
    batch_size: int = 512,
) -> np.ndarray:
    """
    Runs the RegimeAutoencoder encoder on all windows in batches.

    Args:
        model:      trained RegimeAutoencoder
        windows:    (N, seq_len, n_features) float tensor — raw, unnormalized
        batch_size: inference batch size

    Returns:
        latents: (N, latent_size) numpy array
    """
    model.eval()

    mean = windows.mean(dim=(0, 1), keepdim=True)
    std  = windows.std(dim=(0, 1),  keepdim=True) + 1e-8
    windows_norm = (windows - mean) / std

    latents = []
    with torch.no_grad():
        for i in range(0, len(windows_norm), batch_size):
            batch = windows_norm[i : i + batch_size]
            z = model.encode(batch)
            latents.append(z.cpu().numpy())

    latents = np.vstack(latents)
    print(f"Extracted latents: {latents.shape}")
    return latents


# ---------------------------------------------------------------------------
# 2. Rank-based cluster labeling
# ---------------------------------------------------------------------------

def label_clusters_by_rank(cluster_stats: dict) -> dict:
    """
    Assigns bull/bear/volatile/sideways by ranking clusters.
    Guarantees all 4 labels are used exactly once regardless of the
    return distribution (handles all-positive or all-negative markets).

    Logic:
        volatile  = highest std  (isolate noise first)
        bull      = highest mean among remaining 3
        bear      = lowest  mean among remaining 3
        sideways  = the one left over
    """
    ids = list(cluster_stats.keys())
    assert len(ids) == 4, f"Expected 4 clusters, got {len(ids)}"

    volatile_id = max(ids, key=lambda c: cluster_stats[c]["std"])
    remaining   = [c for c in ids if c != volatile_id]

    bull_id   = max(remaining, key=lambda c: cluster_stats[c]["mean"])
    remaining = [c for c in remaining if c != bull_id]

    bear_id     = min(remaining, key=lambda c: cluster_stats[c]["mean"])
    sideways_id = [c for c in remaining if c != bear_id][0]

    return {
        volatile_id: "volatile",
        bull_id:     "bull",
        bear_id:     "bear",
        sideways_id: "sideways",
    }


# ---------------------------------------------------------------------------
# 3. Fit KMeans and produce cluster_map + saved labels parquet
# ---------------------------------------------------------------------------

def fit_regimes(
    model,
    windows: torch.Tensor,
    metadata: list,
    df: pd.DataFrame,
    n_clusters: int = None,
    save_path: Path = None,
) -> dict:
    """
    Fits KMeans on the latent space and produces a cluster-to-label map.
    Scores each cluster using the mean intra-window daily log return so
    that labels reflect actual price dynamics of the window, not just
    the single forward return at the end date.

    Args:
        model:      trained RegimeAutoencoder
        windows:    (N, seq_len, n_features) tensor from make_regime_windows
        metadata:   list of (ticker, end_date) from make_regime_windows
        df:         cleaned price DataFrame with a "log_return" column
        n_clusters: number of KMeans clusters (defaults to REGIME["n_clusters"])
        save_path:  if provided, saves regime labels parquet here

    Returns:
        cluster_map: {cluster_int -> regime_label_string}
    """
    n_clusters = n_clusters or REGIME["n_clusters"]
    seq_len    = REGIME["sequence_length"]

    # Step 1: encode
    latents = extract_latents(model, windows)

    # Step 2: scale latents (KMeans is distance-based)
    scaler         = StandardScaler()
    latents_scaled = scaler.fit_transform(latents)

    # Step 3: fit KMeans
    print(f"Fitting KMeans with {n_clusters} clusters on {len(latents)} windows...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(latents_scaled)

    # Step 4: score clusters on intra-window mean daily log return
    # Build (ticker, date) -> log_return lookup
    df_copy = df.copy()
    df_copy["date"] = pd.to_datetime(df_copy["date"])
    log_ret_lookup = (
        df_copy.set_index(["ticker", "date"])["log_return"]
        .to_dict()
    )

    # Pre-build per-ticker sorted int64 timestamp arrays for fast searchsorted.
    # np.searchsorted requires a homogeneous comparable array — int64 nanoseconds
    # avoids the Timestamp vs int comparison TypeError.
    ticker_dates_ns_map = {}   # ticker -> sorted int64 ns array
    ticker_dates_ts_map = {}   # ticker -> sorted Timestamp array (for lookup)

    for ticker, grp in df_copy.groupby("ticker"):
        sorted_dates = grp["date"].sort_values().values          # numpy datetime64
        ticker_dates_ns_map[ticker] = sorted_dates.astype("int64")  # ns since epoch
        ticker_dates_ts_map[ticker] = pd.to_datetime(sorted_dates)  # Timestamps

    cluster_returns = {c: [] for c in range(n_clusters)}

    for i, (ticker, end_date) in enumerate(metadata):
        end_ts_ns = pd.Timestamp(end_date).value  # int64 nanoseconds

        dates_ns = ticker_dates_ns_map.get(ticker)
        dates_ts = ticker_dates_ts_map.get(ticker)

        if dates_ns is None or len(dates_ns) == 0:
            continue

        # searchsorted on int64 — no type mismatch
        end_pos   = int(np.searchsorted(dates_ns, end_ts_ns, side="right")) - 1
        start_pos = max(0, end_pos - seq_len + 1)
        window_ts = dates_ts[start_pos : end_pos + 1]

        rets = [
            log_ret_lookup.get((ticker, ts), np.nan)
            for ts in window_ts
        ]
        rets = [r for r in rets if not np.isnan(r)]

        if rets:
            cluster_returns[labels[i]].append(float(np.mean(rets)))

    cluster_stats = {}
    for c in range(n_clusters):
        rets = cluster_returns[c]
        cluster_stats[c] = {
            "mean": float(np.mean(rets)) if rets else 0.0,
            "std":  float(np.std(rets))  if rets else 0.0,
        }

    # Step 5: rank-based labeling — always 4 distinct labels
    cluster_map = label_clusters_by_rank(cluster_stats)

    print("Cluster assignments:")
    for c, label in cluster_map.items():
        stats = cluster_stats[c]
        count = int((labels == c).sum())
        print(f"  Cluster {c} -> {label:10s}  "
              f"mean_ret={stats['mean']:+.7f}  std={stats['std']:.7f}  n={count}")

    # Step 6: build and save the labeled parquet
    rows = [
        {
            "ticker":  ticker,
            "date":    pd.Timestamp(end_date),
            "cluster": int(labels[i]),
            "regime":  cluster_map[int(labels[i])],
        }
        for i, (ticker, end_date) in enumerate(metadata)
    ]
    regime_df = pd.DataFrame(rows)

    out_path = save_path or (DATA_DIR / "regime_labels.parquet")
    regime_df.to_parquet(out_path, index=False)
    print(f"Saved regime labels to {out_path}  ({len(regime_df)} rows)")

    return cluster_map


# ---------------------------------------------------------------------------
# 4. Attach regime labels onto the main DataFrame
# ---------------------------------------------------------------------------

def attach_regime_labels(
    df: pd.DataFrame,
    cluster_map: dict,
    regime_path: Path = None,
) -> pd.DataFrame:
    """
    Merges the saved regime_labels.parquet onto the main price DataFrame.
    Reads the parquet saved by fit_regimes — does NOT re-run KMeans.

    Args:
        df:          cleaned price DataFrame
        cluster_map: output of fit_regimes (used only for logging)
        regime_path: path to regime_labels.parquet (defaults to DATA_DIR)

    Returns:
        df with a new "regime" column. Unmatched rows get "unknown".
    """
    path = regime_path or (DATA_DIR / "regime_labels.parquet")
    if not path.exists():
        raise FileNotFoundError(
            f"regime_labels.parquet not found at {path}. "
            "Run fit_regimes() first."
        )

    regime_df = pd.read_parquet(path)[["ticker", "date", "regime"]]

    df = df.copy()
    df["date"]        = pd.to_datetime(df["date"])
    regime_df["date"] = pd.to_datetime(regime_df["date"])

    merged = df.merge(regime_df, on=["ticker", "date"], how="left")
    merged["regime"] = merged["regime"].fillna("unknown")

    coverage = (merged["regime"] != "unknown").mean()
    print(f"Regime labels attached. Coverage: {coverage:.1%} of rows labeled")
    print(f"Regime distribution:\n{merged['regime'].value_counts().to_string()}")

    return merged


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data import (
        _make_mock_ohlcv, add_technical_indicators,
        add_target, add_time_idx, clean_and_normalize,
    )
    from model import RegimeAutoencoder, make_regime_windows

    print("Building mock dataset...")
    tickers = ["AAPL", "MSFT", "XOM", "JPM", "NEE",
               "NVDA", "GS", "LMT", "CVX", "RTX"]
    df = _make_mock_ohlcv(tickers, "2020-01-01", "2023-12-31")
    df = add_technical_indicators(df)
    df = add_target(df)
    df = add_time_idx(df)
    df = clean_and_normalize(df)

    feature_cols = OHLCV_COLS + INDICATOR_COLS
    windows, metadata = make_regime_windows(df, feature_cols)

    print("\nRunning regime pipeline with untrained model (sanity check only)...")
    ae_model = RegimeAutoencoder(input_size=len(feature_cols))

    cluster_map = fit_regimes(ae_model, windows, metadata, df)
    df_labeled  = attach_regime_labels(df, cluster_map)

    print(f"\nFinal DataFrame shape: {df_labeled.shape}")
    print(df_labeled[["ticker", "date", "regime"]].head(10).to_string())