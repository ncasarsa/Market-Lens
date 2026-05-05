"""
regime.py
---------
Post-training pipeline for market regime detection.

After train_regime_ae() produces a trained RegimeAutoencoder, this module:
    1. Encodes all windows into latent vectors
    2. Fits KMeans on those vectors
    3. Labels each cluster semantically (bull / bear / volatile / sideways)
       based on mean return and return variance of the windows in each cluster
    4. Merges a "regime" string column back onto the main price DataFrame

Usage:
    from regime import fit_regimes, attach_regime_labels

    # After training
    ae_model, windows, metadata = train_regime_ae(df)
    cluster_map = fit_regimes(ae_model, windows, metadata, df)
    df = attach_regime_labels(df, ae_model, cluster_map)
    df.to_parquet("data/dataset_with_regimes.parquet", index=False)

Regime label semantics:
    bull       high mean return, low variance
    bear       low (negative) mean return, low variance
    volatile   high variance regardless of direction
    sideways   near-zero mean return, low variance
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
        model:      trained RegimeAutoencoder (output of train_regime_ae)
        windows:    (N, seq_len, n_features) float tensor — raw, unnormalized
        batch_size: inference batch size

    Returns:
        latents: (N, latent_size) numpy array
    """
    model.eval()

    # Normalize the same way trainer.py does before encoding
    mean = windows.mean(dim=(0, 1), keepdim=True)
    std  = windows.std(dim=(0, 1), keepdim=True) + 1e-8
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
# 2. Fit KMeans and label clusters semantically
# ---------------------------------------------------------------------------

def _label_cluster(mean_ret: float, std_ret: float, vol_threshold: float) -> str:
    """
    Assigns a semantic label to a cluster given its return statistics.

    Labeling logic:
        volatile  — std above the cross-cluster median (high uncertainty)
        bull      — mean return in top half of non-volatile clusters
        bear      — mean return in bottom half of non-volatile clusters
        sideways  — everything else (near-zero drift, low volatility)
    """
    if std_ret >= vol_threshold:
        return "volatile"
    if mean_ret > 0.0003:        # roughly +7.5% annualized threshold
        return "bull"
    if mean_ret < -0.0003:       # roughly -7.5% annualized threshold
        return "bear"
    return "sideways"


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

    Args:
        model:      trained RegimeAutoencoder
        windows:    (N, seq_len, n_features) tensor from make_regime_windows
        metadata:   list of (ticker, end_date) from make_regime_windows
        df:         cleaned price DataFrame — used to look up returns per window
        n_clusters: number of KMeans clusters (defaults to REGIME["n_clusters"])
        save_path:  if provided, saves regime labels parquet here

    Returns:
        cluster_map: dict mapping cluster int -> regime label string
                     e.g. {0: "bull", 1: "bear", 2: "volatile", 3: "sideways"}
    """
    n_clusters = n_clusters or REGIME["n_clusters"]
    seq_len    = REGIME["sequence_length"]

    # Step 1: encode
    latents = extract_latents(model, windows)

    # Step 2: scale latents before clustering (KMeans is distance-based)
    scaler  = StandardScaler()
    latents_scaled = scaler.fit_transform(latents)

    # Step 3: fit KMeans
    print(f"Fitting KMeans with {n_clusters} clusters on {len(latents)} windows...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(latents_scaled)

    # Step 4: compute per-cluster return statistics to assign semantic labels
    # Build a lookup: (ticker, date) -> target (log return) from df
    ret_lookup = (
        df.set_index(["ticker", pd.to_datetime(df["date"])])[TARGET_COL]
        .to_dict()
    )

    cluster_returns = {c: [] for c in range(n_clusters)}
    for i, (ticker, end_date) in enumerate(metadata):
        end_date_ts = pd.Timestamp(end_date)
        key = (ticker, end_date_ts)
        ret = ret_lookup.get(key, np.nan)
        if not np.isnan(ret):
            cluster_returns[labels[i]].append(ret)

    cluster_stats = {}
    for c in range(n_clusters):
        rets = cluster_returns[c]
        if len(rets) == 0:
            cluster_stats[c] = {"mean": 0.0, "std": 0.0}
        else:
            cluster_stats[c] = {
                "mean": float(np.mean(rets)),
                "std":  float(np.std(rets)),
            }

    # Volatility threshold = median std across clusters
    vol_threshold = float(np.median([s["std"] for s in cluster_stats.values()]))

    cluster_map = {}
    for c, stats in cluster_stats.items():
        cluster_map[c] = _label_cluster(stats["mean"], stats["std"], vol_threshold)

    print("Cluster assignments:")
    for c, label in cluster_map.items():
        stats = cluster_stats[c]
        count = int((labels == c).sum())
        print(f"  Cluster {c} -> {label:10s}  "
              f"mean_ret={stats['mean']:+.5f}  std={stats['std']:.5f}  n={count}")

    # Step 5: build a (ticker, date, regime) DataFrame and optionally save
    rows = []
    for i, (ticker, end_date) in enumerate(metadata):
        rows.append({
            "ticker": ticker,
            "date":   pd.Timestamp(end_date),
            "cluster": int(labels[i]),
            "regime":  cluster_map[int(labels[i])],
        })

    regime_df = pd.DataFrame(rows)

    out_path = save_path or (DATA_DIR / "regime_labels.parquet")
    regime_df.to_parquet(out_path, index=False)
    print(f"Saved regime labels to {out_path}  ({len(regime_df)} rows)")

    return cluster_map


# ---------------------------------------------------------------------------
# 3. Attach regime labels onto the main DataFrame
# ---------------------------------------------------------------------------

def attach_regime_labels(
    df: pd.DataFrame,
    model,
    cluster_map: dict,
    windows: torch.Tensor = None,
    metadata: list = None,
    regime_path: Path = None,
) -> pd.DataFrame:
    """
    Merges a "regime" string column onto the main price DataFrame.

    Each row in df gets the regime label of the most recent window
    whose end_date matches that row's (ticker, date).

    If windows and metadata are provided, re-encodes and re-clusters inline.
    Otherwise, loads from regime_path (or the default DATA_DIR location).

    Args:
        df:          cleaned price DataFrame
        model:       trained RegimeAutoencoder
        cluster_map: output of fit_regimes
        windows:     optional — tensor from make_regime_windows
        metadata:    optional — list from make_regime_windows
        regime_path: optional — path to saved regime_labels.parquet

    Returns:
        df with a new "regime" column (str).
        Rows with no matching window are labeled "unknown".
    """
    # Load or build the regime labels DataFrame
    if windows is not None and metadata is not None:
        latents = extract_latents(model, windows)
        scaler  = StandardScaler()
        latents_scaled = scaler.fit_transform(latents)
        kmeans = KMeans(n_clusters=len(cluster_map), random_state=42, n_init=10)
        raw_labels = kmeans.fit_predict(latents_scaled)

        rows = [
            {
                "ticker": ticker,
                "date":   pd.Timestamp(end_date),
                "regime": cluster_map[int(raw_labels[i])],
            }
            for i, (ticker, end_date) in enumerate(metadata)
        ]
        regime_df = pd.DataFrame(rows)
    else:
        path = regime_path or (DATA_DIR / "regime_labels.parquet")
        if not path.exists():
            raise FileNotFoundError(
                f"regime_labels.parquet not found at {path}. "
                "Run fit_regimes() first or pass windows and metadata."
            )
        regime_df = pd.read_parquet(path)[["ticker", "date", "regime"]]

    # Merge onto main df — left join so all price rows are preserved
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    regime_df["date"] = pd.to_datetime(regime_df["date"])

    merged = df.merge(regime_df[["ticker", "date", "regime"]], on=["ticker", "date"], how="left")
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

    # Use an untrained model — labels will be random but pipeline logic is verified
    print("\nRunning regime pipeline with untrained model (sanity check only)...")
    ae_model = RegimeAutoencoder(input_size=len(feature_cols))

    cluster_map = fit_regimes(ae_model, windows, metadata, df)

    df_labeled = attach_regime_labels(df, ae_model, cluster_map, windows, metadata)

    print(f"\nFinal DataFrame shape: {df_labeled.shape}")
    print(df_labeled[["ticker", "date", "regime"]].head(10).to_string())