"""
model.py
--------
Defines two models:

1. build_tft() — wraps PyTorch Forecasting's TemporalFusionTransformer.
   Takes a TimeSeriesDataSet and config dict, returns a ready-to-train TFT.

2. RegimeAutoencoder — LSTM autoencoder for unsupervised market regime
   detection. Encodes fixed-length windows into a latent vector, which
   is then clustered (KMeans) into regime labels.

3. build_timeseries_dataset() — constructs the TimeSeriesDataSet from
   the cleaned DataFrame produced by data.py.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, RMSE, MAPE, SMAPE, QuantileLoss
from pytorch_forecasting.data import GroupNormalizer

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    TFT as TFT_CFG,
    TRAIN,
    STATIC_CATEGORICALS,
    STATIC_REALS,
    TIME_VARYING_KNOWN_REALS,
    TIME_VARYING_UNKNOWN_REALS,
    TARGET_COL,
    GROUP_COL,
    TIME_COL,
    VAL_START,
    TEST_START,
    REGIME,
)


# ---------------------------------------------------------------------------
# 1. TimeSeriesDataSet builder
# ---------------------------------------------------------------------------

def build_timeseries_dataset(
    df: pd.DataFrame,
    max_encoder_length: int = None,
    max_prediction_length: int = None,
    cutoff: str = VAL_START,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    """
    Splits df into train / val TimeSeriesDataSets.

    Args:
        df:                     cleaned DataFrame from data.py
        max_encoder_length:     lookback window (defaults to TFT_CFG value)
        max_prediction_length:  forecast horizon (defaults to TFT_CFG value)
        cutoff:                 date string — rows before this are train,
                                on/after are val

    Returns:
        (train_dataset, val_dataset)
    """
    max_encoder_length    = max_encoder_length    or TFT_CFG["max_encoder_length"]
    max_prediction_length = max_prediction_length or TFT_CFG["max_prediction_length"]

    # Convert cutoff date to time_idx
    cutoff_dt  = pd.to_datetime(cutoff)
    cutoff_idx = int(df.loc[df["date"] >= cutoff_dt, TIME_COL].min())

    # Ensure sector and ticker are strings (PF categorical requirement)
    df = df.copy()
    df["ticker"] = df["ticker"].astype(str)
    df["sector"]  = df["sector"].astype(str)

    train_dataset = TimeSeriesDataSet(
        df[df[TIME_COL] < cutoff_idx],
        time_idx                        = TIME_COL,
        target                          = TARGET_COL,
        group_ids                       = [GROUP_COL],
        max_encoder_length              = max_encoder_length,
        min_encoder_length              = max_encoder_length,  # fix histogram OOB error
        max_prediction_length           = max_prediction_length,
        static_categoricals             = STATIC_CATEGORICALS,
        static_reals                    = STATIC_REALS,
        time_varying_known_reals        = TIME_VARYING_KNOWN_REALS,
        time_varying_unknown_reals      = TIME_VARYING_UNKNOWN_REALS + [TARGET_COL],
        target_normalizer               = GroupNormalizer(
                                            groups=[GROUP_COL],
                                            transformation=None,
                                          ),
        add_relative_time_idx           = False,
        add_target_scales               = True,
        add_encoder_length              = True,
        allow_missing_timesteps         = True,
    )

    # Val dataset shares the same parameters as train (critical for PF)
    val_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset,
        df[df[TIME_COL] >= cutoff_idx],
        predict=False,
        stop_randomization=True,
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset:   {len(val_dataset)} samples")
    return train_dataset, val_dataset


# ---------------------------------------------------------------------------
# 2. TFT builder
# ---------------------------------------------------------------------------
class MarketTFT(TemporalFusionTransformer):
    """Disables PF's broken interpretation logging during validation."""
    def _log_interpretation(self, out):
        return {}

    def log_interpretation(self, outputs):
        pass

def build_tft(
    train_dataset: TimeSeriesDataSet,
    cfg: dict = None,
) -> MarketTFT:
    """
    Instantiates a MarketTFT from a TimeSeriesDataSet.

    Args:
        train_dataset:  TimeSeriesDataSet (used to infer embedding sizes etc.)
        cfg:            hyperparameter dict (defaults to config.TFT)

    Returns:
        MarketTFT (not yet trained)
    """
    cfg = cfg or TFT_CFG

    model = MarketTFT.from_dataset(
        train_dataset,
        learning_rate           = cfg["learning_rate"],
        hidden_size             = cfg["hidden_size"],
        attention_head_size     = cfg["attention_head_size"],
        dropout                 = cfg["dropout"],
        hidden_continuous_size  = cfg["hidden_size"] // 2,
        lstm_layers             = cfg["lstm_layers"],
        loss                    = QuantileLoss(),          # probabilistic output
        logging_metrics         = nn.ModuleList([MAE(), RMSE(), MAPE(), SMAPE()]),
        reduce_on_plateau_patience = 100,
        log_interval            = 10,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TFT initialized — {total_params:,} trainable parameters")
    return model


# ---------------------------------------------------------------------------
# 3. LSTM Autoencoder (regime detection)
# ---------------------------------------------------------------------------

class RegimeAutoencoder(nn.Module):
    """
    LSTM autoencoder that compresses fixed-length multivariate windows
    into a low-dimensional latent vector for regime clustering.

    Architecture:
        Encoder: LSTM → takes (seq_len, input_size) → produces latent vector
        Decoder: LSTM → reconstructs input from latent vector

    Usage:
        model = RegimeAutoencoder()
        latent = model.encode(x)        # (batch, latent_size)
        recon  = model(x)               # (batch, seq_len, input_size)
        loss   = F.mse_loss(recon, x)
    """

    def __init__(
        self,
        input_size:      int = None,
        hidden_size:     int = None,
        latent_size:     int = None,
        num_layers:      int = None,
        dropout:         float = None,
    ):
        super().__init__()

        input_size  = input_size  or REGIME["input_size"]
        hidden_size = hidden_size or REGIME["hidden_size"]
        latent_size = latent_size or REGIME["latent_size"]
        num_layers  = num_layers  or REGIME["num_layers"]
        dropout     = dropout     or REGIME["dropout"]

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers  = num_layers

        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.encoder_fc = nn.Linear(hidden_size, latent_size)

        # Decoder
        self.decoder_fc = nn.Linear(latent_size, hidden_size)
        self.decoder_lstm = nn.LSTM(
            input_size  = hidden_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.output_fc = nn.Linear(hidden_size, input_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            latent: (batch, latent_size)
        """
        _, (h_n, _) = self.encoder_lstm(x)
        # h_n: (num_layers, batch, hidden_size) — take last layer
        latent = self.encoder_fc(h_n[-1])
        return latent

    def decode(self, latent: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Args:
            latent:  (batch, latent_size)
            seq_len: length of sequence to reconstruct
        Returns:
            recon: (batch, seq_len, input_size)
        """
        # Expand latent into seq_len repeated hidden states
        h = self.decoder_fc(latent)                        # (batch, hidden_size)
        h = h.unsqueeze(1).repeat(1, seq_len, 1)           # (batch, seq_len, hidden_size)
        out, _ = self.decoder_lstm(h)
        recon = self.output_fc(out)
        return recon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        recon  = self.decode(latent, seq_len=x.size(1))
        return recon


def make_regime_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    sequence_length: int = None,
) -> tuple[torch.Tensor, list]:
    """
    Slides a window over each ticker's time series and stacks into a
    3D tensor for the RegimeAutoencoder.

    Args:
        df:             cleaned DataFrame from data.py
        feature_cols:   columns to use as input features
        sequence_length: window size (defaults to REGIME["sequence_length"])

    Returns:
        windows:  (N, seq_len, n_features) float tensor
        metadata: list of (ticker, end_date) for each window
    """
    seq_len = sequence_length or REGIME["sequence_length"]
    windows, metadata = [], []

    for ticker, group in df.groupby("ticker"):
        group = group.sort_values("date")
        feats = group[feature_cols].values.astype(np.float32)
        dates = group["date"].values

        for i in range(len(feats) - seq_len + 1):
            windows.append(feats[i : i + seq_len])
            metadata.append((ticker, dates[i + seq_len - 1]))

    windows_tensor = torch.tensor(np.stack(windows), dtype=torch.float32)
    print(f"Regime windows: {windows_tensor.shape}  "
          f"(N={windows_tensor.shape[0]}, seq={seq_len}, features={windows_tensor.shape[2]})")
    return windows_tensor, metadata


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Use mock data so this runs offline
    from data import (
        _make_mock_ohlcv, add_technical_indicators,
        add_target, add_time_idx, clean_and_normalize,
    )
    from config import INDICATOR_COLS, OHLCV_COLS

    print("Building mock dataset...")
    tickers = ["AAPL", "MSFT", "XOM", "JPM", "NEE"]
    df = _make_mock_ohlcv(tickers, "2020-01-01", "2023-12-31")
    df = add_technical_indicators(df)
    df = add_target(df)
    df = add_time_idx(df)
    df = clean_and_normalize(df)

    print("\n--- TimeSeriesDataSet ---")
    train_ds, val_ds = build_timeseries_dataset(df, cutoff="2023-01-01")

    print("\n--- TFT ---")
    model = build_tft(train_ds)
    print(model)

    print("\n--- RegimeAutoencoder ---")
    regime_model = RegimeAutoencoder()
    feature_cols = OHLCV_COLS + INDICATOR_COLS
    windows, meta = make_regime_windows(df, feature_cols)

    # Forward pass test
    batch = windows[:8]
    recon = regime_model(batch)
    latent = regime_model.encode(batch)
    print(f"Input shape:   {batch.shape}")
    print(f"Recon shape:   {recon.shape}")
    print(f"Latent shape:  {latent.shape}")
    print("RegimeAutoencoder forward pass OK")