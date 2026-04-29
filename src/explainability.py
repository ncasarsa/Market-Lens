"""
explainability.py
-----------------
Post-hoc explainability for the trained TFT model.

1. attention_heatmap()     — extracts and plots temporal attention weights
                             over the past N trading days per ticker
2. variable_importance()   — plots encoder/decoder/static variable selection
                             weights from TFT's built-in interpret_output()
3. shap_waterfall()        — GradientExplainer SHAP values for a single
                             prediction, plotted as a waterfall chart
4. explain_prediction()    — convenience wrapper: runs all three for one
                             ticker on one date

Usage (from project root):
    python src/explainability.py --ticker AAPL --date 2024-06-01
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import shap

from pytorch_forecasting import TimeSeriesDataSet

from config import (
    TIME_VARYING_UNKNOWN_REALS,
    STATIC_CATEGORICALS,
    TARGET_COL,
    GROUP_COL,
    TIME_COL,
    CHECKPOINT_PATH,
    VAL_START,
    TFT as TFT_CFG,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path: str = None):
    """Load MarketTFT from checkpoint."""
    from model import MarketTFT
    ckpt = checkpoint_path or str(CHECKPOINT_PATH.parent / "tft_best.ckpt")
    model = MarketTFT.load_from_checkpoint(ckpt)
    model.eval()
    return model


def _get_feature_names(model) -> dict:
    """
    Extract feature name lists from a trained TFT model.
    Returns dict with keys: encoder, decoder, static_cats, static_reals.
    """
    hparams = model.hparams
    return {
        "encoder": (
            list(hparams.get("time_varying_reals_encoder", [])) +
            list(hparams.get("time_varying_categoricals_encoder", []))
        ),
        "decoder": (
            list(hparams.get("time_varying_reals_decoder", [])) +
            list(hparams.get("time_varying_categoricals_decoder", []))
        ),
        "static_cats":  list(hparams.get("static_categoricals", [])),
        "static_reals": list(hparams.get("static_reals", [])),
    }


def _build_single_ticker_loader(
    df: pd.DataFrame,
    ticker: str,
    train_dataset: TimeSeriesDataSet,
    batch_size: int = 64,
):
    """Build a dataloader filtered to a single ticker for inference."""
    ticker_df = df[df[GROUP_COL] == ticker].copy()
    ds = TimeSeriesDataSet.from_dataset(
        train_dataset,
        ticker_df,
        predict=False,
        stop_randomization=True,
        min_encoder_length=TFT_CFG["max_encoder_length"],
    )
    return ds.to_dataloader(train=False, batch_size=batch_size, num_workers=0)


# ---------------------------------------------------------------------------
# 1. Attention Heatmap
# ---------------------------------------------------------------------------

def attention_heatmap(
    model,
    predictions,
    feature_names: list[str],
    ticker: str = None,
    n_samples: int = 30,
    save_path: str = None,
) -> plt.Figure:
    """
    Plots a heatmap of temporal self-attention weights over the encoder
    window. Rows = prediction samples, Cols = encoder timesteps.

    Args:
        model:         trained MarketTFT
        predictions:   output from model.predict(..., mode='raw', return_x=True)
        feature_names: list of encoder feature names
        ticker:        ticker label for plot title
        n_samples:     number of prediction samples to show
        save_path:     if provided, saves figure to this path

    Returns:
        matplotlib Figure
    """
    # attention shape: (N, n_heads, pred_len, encoder_len)
    attention = predictions.output.attention.detach().cpu()

    # Average over heads and prediction horizon → (N, encoder_len)
    attn_avg = attention.mean(dim=(1, 2))[:n_samples]  # (n_samples, encoder_len)

    fig, ax = plt.subplots(figsize=(14, max(4, n_samples // 4)))
    im = ax.imshow(
        attn_avg.numpy(),
        aspect="auto",
        cmap="YlOrRd",
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label="Attention Weight")
    ax.set_xlabel("Encoder Timestep (0 = oldest, right = most recent)")
    ax.set_ylabel("Prediction Sample")
    title = f"TFT Temporal Attention Weights"
    if ticker:
        title += f" — {ticker}"
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 2. Variable Importance
# ---------------------------------------------------------------------------

def variable_importance(
    model,
    predictions,
    encoder_feature_names: list[str],
    decoder_feature_names: list[str] = None,
    static_feature_names: list[str] = None,
    top_n: int = 20,
    save_path: str = None,
) -> plt.Figure:
    """
    Plots TFT variable selection weights for encoder, decoder, and static
    inputs as horizontal bar charts.

    Args:
        model:                  trained MarketTFT
        predictions:            output from model.predict(..., mode='raw')
        encoder_feature_names:  list of encoder input names
        decoder_feature_names:  list of decoder input names (optional)
        static_feature_names:   list of static input names (optional)
        top_n:                  number of top features to show per panel
        save_path:              if provided, saves figure to this path

    Returns:
        matplotlib Figure
    """
    interpretation = model.interpret_output(
        predictions.output,
        reduction="mean",
    )

    # PF appends relative_time_idx internally
    enc_names = encoder_feature_names + ["relative_time_idx"]
    enc_imp   = interpretation["encoder_variables"].cpu().numpy()
    enc_series = pd.Series(
        enc_imp, index=enc_names[:len(enc_imp)]
    ).sort_values(ascending=True).tail(top_n)

    n_panels = 1
    dec_series = static_series = None

    dec_imp = interpretation.get("decoder_variables")
    if dec_imp is not None and decoder_feature_names:
        dec_series = pd.Series(
            dec_imp.cpu().numpy(),
            index=(decoder_feature_names + ["relative_time_idx"])[:len(dec_imp)]
        ).sort_values(ascending=True).tail(top_n)
        n_panels += 1

    static_imp = interpretation.get("static_variables")
    if static_imp is not None and static_feature_names:
        static_series = pd.Series(
            static_imp.cpu().numpy(),
            index=static_feature_names[:len(static_imp)]
        ).sort_values(ascending=True)
        n_panels += 1

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 8))
    if n_panels == 1:
        axes = [axes]

    # Encoder
    axes[0].barh(enc_series.index, enc_series.values, color="steelblue")
    axes[0].set_title(f"Encoder Variable Importance (Top {top_n})")
    axes[0].set_xlabel("Importance")
    axes[0].axvline(0, color="black", linewidth=0.5)

    panel = 1
    if dec_series is not None:
        axes[panel].barh(dec_series.index, dec_series.values, color="seagreen")
        axes[panel].set_title(f"Decoder Variable Importance (Top {top_n})")
        axes[panel].set_xlabel("Importance")
        panel += 1

    if static_series is not None:
        axes[panel].barh(static_series.index, static_series.values, color="darkorange")
        axes[panel].set_title("Static Variable Importance")
        axes[panel].set_xlabel("Importance")

    plt.suptitle("TFT Variable Selection Weights", fontsize=14, y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 3. SHAP Waterfall
# ---------------------------------------------------------------------------

def shap_waterfall(
    model,
    train_loader,
    val_loader,
    feature_names: list[str],
    sample_idx: int = 0,
    n_background: int = 100,
    save_path: str = None,
) -> plt.Figure:
    """
    Computes GradientExplainer SHAP values for a single prediction and
    plots them as a waterfall chart showing each feature's contribution.

    Args:
        model:          trained MarketTFT (CPU or GPU)
        train_loader:   DataLoader used as SHAP background distribution
        val_loader:     DataLoader to explain predictions from
        feature_names:  encoder feature names
        sample_idx:     which val sample to explain (0-indexed)
        n_background:   number of background samples for GradientExplainer
        save_path:      if provided, saves figure to this path

    Returns:
        matplotlib Figure
    """
    device = next(model.parameters()).device
    model.eval()

    # Collect background samples from train loader
    bg_encoder_inputs = []
    count = 0
    for batch in train_loader:
        x, _ = batch
        enc = x["encoder_cont"].to(device)
        bg_encoder_inputs.append(enc)
        count += enc.shape[0]
        if count >= n_background:
            break
    background = torch.cat(bg_encoder_inputs, dim=0)[:n_background]

    # Get the specific val sample to explain
    val_batches = []
    for batch in val_loader:
        x, _ = batch
        val_batches.append(x["encoder_cont"].to(device))
    val_enc = torch.cat(val_batches, dim=0)
    sample = val_enc[sample_idx:sample_idx + 1]  # (1, seq_len, n_features)

    # Wrap model to accept raw encoder_cont tensor
    class EncoderWrapper(torch.nn.Module):
        def __init__(self, tft_model, sample_batch_x):
            super().__init__()
            self.tft   = tft_model
            self.ref_x = sample_batch_x  # store full batch_x as reference

        def forward(self, encoder_cont):
            # Rebuild a minimal batch_x with swapped encoder_cont
            import copy
            batch_x = copy.deepcopy(self.ref_x)
            batch_x["encoder_cont"] = encoder_cont
            out = self.tft(batch_x)
            # Return median quantile (index 3) prediction
            return out["prediction"][:, :, 3]  # (batch, pred_len)

    # Get a reference batch_x
    for batch in val_loader:
        ref_x, _ = batch
        ref_x = {k: v.to(device) for k, v in ref_x.items()}
        break

    wrapper = EncoderWrapper(model, ref_x)

    # GradientExplainer
    explainer = shap.GradientExplainer(wrapper, background)
    shap_values = explainer.shap_values(sample)  # list or array

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Average SHAP over sequence length → (n_features,)
    shap_mean = np.abs(shap_values[0]).mean(axis=0)

    shap_series = pd.Series(shap_mean, index=feature_names[:len(shap_mean)])
    shap_series = shap_series.sort_values(ascending=False).head(20)

    # Waterfall-style bar chart
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#d73027" if v > 0 else "#4575b4" for v in shap_series.values]
    ax.barh(
        shap_series.index[::-1],
        shap_series.values[::-1],
        color=colors[::-1],
    )
    ax.set_title(f"SHAP Feature Attribution — Sample {sample_idx}\n"
                 f"(mean |SHAP| over 60-day encoder window)")
    ax.set_xlabel("|SHAP value| — average impact on prediction")
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 4. Convenience wrapper
# ---------------------------------------------------------------------------

def explain_prediction(
    df: pd.DataFrame,
    train_dataset: TimeSeriesDataSet,
    ticker: str,
    checkpoint_path: str = None,
    save_dir: str = None,
) -> dict:
    """
    Runs all three explainability methods for a given ticker and returns
    a dict of figures.

    Args:
        df:               full cleaned DataFrame
        train_dataset:    TimeSeriesDataSet used during training
        ticker:           ticker symbol to explain
        checkpoint_path:  path to TFT checkpoint (defaults to config path)
        save_dir:         if provided, saves all figures here

    Returns:
        dict with keys: 'attention', 'importance', 'shap'
    """
    from model import build_timeseries_dataset
    from torch.utils.data import DataLoader

    print(f"Explaining predictions for {ticker}...")

    model = _load_model(checkpoint_path)
    feature_names = _get_feature_names(model)

    # Full val loader
    _, val_ds = build_timeseries_dataset(df, cutoff=VAL_START)
    val_loader = val_ds.to_dataloader(train=False, batch_size=64, num_workers=0)

    # Train loader (for SHAP background)
    train_ds, _ = build_timeseries_dataset(df, cutoff=VAL_START)
    train_loader = train_ds.to_dataloader(train=True, batch_size=64, num_workers=0)

    # Run predictions
    print("  Running predictions...")
    predictions = model.predict(val_loader, mode="raw", return_x=True)

    enc_names = feature_names["encoder"]

    # Attention heatmap
    print("  Computing attention heatmap...")
    attn_save = os.path.join(save_dir, f"{ticker}_attention.png") if save_dir else None
    fig_attn = attention_heatmap(model, predictions, enc_names, ticker=ticker,
                                  save_path=attn_save)

    # Variable importance
    print("  Computing variable importance...")
    imp_save = os.path.join(save_dir, f"{ticker}_importance.png") if save_dir else None
    static_names = feature_names["static_cats"] + feature_names["static_reals"]
    fig_imp = variable_importance(
        model, predictions, enc_names,
        static_feature_names=static_names if static_names else None,
        save_path=imp_save,
    )

    # SHAP
    print("  Computing SHAP values (this may take ~1-2 minutes)...")
    shap_save = os.path.join(save_dir, f"{ticker}_shap.png") if save_dir else None
    fig_shap = shap_waterfall(
        model, train_loader, val_loader, enc_names,
        sample_idx=0, n_background=50,
        save_path=shap_save,
    )

    print(f"Done. Figures ready for {ticker}.")
    return {"attention": fig_attn, "importance": fig_imp, "shap": fig_shap}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MarketLens explainability")
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    from data import (
        _make_mock_ohlcv, add_technical_indicators,
        add_target, add_time_idx, clean_and_normalize, build_dataset,
    )
    from model import build_timeseries_dataset
    from config import DATASET_PATH, TICKERS, TRAIN_START, TRAIN_END

    if args.mock:
        tickers = ["AAPL", "MSFT", "XOM", "JPM", "NEE",
                   "NVDA", "BAC", "LMT", "CVX", "DUK"]
        df = _make_mock_ohlcv(tickers, "2018-01-01", "2023-12-31")
        df = add_technical_indicators(df)
        df = add_target(df)
        df = add_time_idx(df)
        df = clean_and_normalize(df)
    elif DATASET_PATH.exists():
        df = pd.read_parquet(DATASET_PATH)
    else:
        df = build_dataset(tickers=TICKERS, start=TRAIN_START, end=TRAIN_END,
                           save_path=str(DATASET_PATH))

    train_ds, _ = build_timeseries_dataset(df, cutoff=VAL_START)

    figs = explain_prediction(
        df, train_ds,
        ticker=args.ticker,
        checkpoint_path=args.checkpoint,
        save_dir=args.save_dir,
    )

    plt.show()