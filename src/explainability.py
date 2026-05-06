"""
explainability.py
-----------------
Post-hoc explainability for the trained TFT model.

1. attention_heatmap()     — extracts temporal attention from encoder_attention
2. variable_importance()   — plots encoder variable selection weights
3. shap_waterfall()        — GradientExplainer SHAP for a single prediction

Usage:
    python src/explainability.py --ticker AAPL
"""

import os
import sys
import argparse
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
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
# EncoderWrapper — module-level so it is always in scope
# ---------------------------------------------------------------------------

class EncoderWrapper(nn.Module):
    """
    Wraps a TFT model to accept a raw encoder_cont tensor as input.
    Used by GradientExplainer which needs a simple Tensor -> Tensor function.

    ref_x must already be trimmed to n_background rows so that
    encoder_cont and decoder_cont sizes match in TFT's forward cat.
    """

    def __init__(self, tft_model, ref_batch_x: dict):
        super().__init__()
        self.tft   = tft_model
        self.ref_x = ref_batch_x  # pre-trimmed to n_background

    def forward(self, encoder_cont: torch.Tensor) -> torch.Tensor:
        bs = encoder_cont.shape[0]
        batch_x = {}
        ref_bs  = self.ref_x["encoder_cont"].shape[0]

        for k, v in self.ref_x.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == ref_bs:
                if v.shape[0] >= bs:
                    batch_x[k] = v[:bs]
                else:
                    repeats = (bs // v.shape[0]) + 1
                    batch_x[k] = v.repeat(repeats, *([1] * (v.dim() - 1)))[:bs]
            else:
                batch_x[k] = v

        batch_x["encoder_cont"] = encoder_cont
        out = self.tft(batch_x)
        return out["prediction"][:, :, 3]   # median quantile (index 3 of 7)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path: str = None):
    from model import MarketTFT
    ckpt = checkpoint_path or str(CHECKPOINT_PATH.parent / "tft_best.ckpt")
    model = MarketTFT.load_from_checkpoint(ckpt)
    model.eval()
    return model


def _get_feature_names(model) -> dict:
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


# ---------------------------------------------------------------------------
# 1. Attention Heatmap
# ---------------------------------------------------------------------------

def attention_heatmap(
    model,
    predictions,
    feature_names: list,
    ticker: str = None,
    n_samples: int = 30,
    save_path: str = None,
) -> plt.Figure:
    """
    Plots a heatmap of temporal self-attention weights.
    Uses encoder_attention directly from the raw output namedtuple
    — shape (N, pred_len, n_heads, encoder_len).
    """
    # encoder_attention: (N, pred_len=1, n_heads=2, encoder_len=60)
    attn = predictions.output.encoder_attention.detach().cpu()

    # Average over pred_len and n_heads → (N, encoder_len)
    attn_avg = attn.mean(dim=(1, 2))[:n_samples]   # (n_samples, encoder_len)

    fig, ax = plt.subplots(figsize=(14, max(4, n_samples // 4)))
    im = ax.imshow(attn_avg.numpy(), aspect="auto", cmap="YlOrRd",
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Attention Weight")
    ax.set_xlabel("Encoder Timestep (0 = oldest, right = most recent)")
    ax.set_ylabel("Prediction Sample")
    title = "TFT Temporal Attention Weights"
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
    encoder_feature_names: list,
    decoder_feature_names: list = None,
    static_feature_names: list = None,
    top_n: int = 20,
    save_path: str = None,
) -> plt.Figure:
    """
    Plots TFT variable selection weights for encoder, decoder, and static inputs.
    """
    interpretation = model.interpret_output(
        predictions.output, reduction="mean"
    )

    enc_names  = encoder_feature_names + ["relative_time_idx"]
    enc_imp    = interpretation["encoder_variables"].cpu().numpy()
    enc_series = (
        pd.Series(enc_imp, index=enc_names[:len(enc_imp)])
        .sort_values(ascending=True)
        .tail(top_n)
    )

    n_panels   = 1
    dec_series = static_series = None

    dec_imp = interpretation.get("decoder_variables")
    if dec_imp is not None and decoder_feature_names:
        dec_series = (
            pd.Series(
                dec_imp.cpu().numpy(),
                index=(decoder_feature_names + ["relative_time_idx"])[:len(dec_imp)]
            )
            .sort_values(ascending=True)
            .tail(top_n)
        )
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
    feature_names: list,
    sample_idx: int = 0,
    n_background: int = 32,
    save_path: str = None,
) -> plt.Figure:
    """
    Computes GradientExplainer SHAP values for a single prediction.

    n_background should be ≤ val_loader batch_size (default 32 is safe).
    ref_x is trimmed to n_background to prevent tensor size mismatch in
    TFT's forward (encoder_cont + decoder_cont cat requires equal batch sizes).
    """
    device = next(model.parameters()).device
    model.eval()

    # Collect background samples from train loader
    bg_inputs = []
    count = 0
    for batch in train_loader:
        x, _ = batch
        enc = x["encoder_cont"].to(device)
        bg_inputs.append(enc)
        count += enc.shape[0]
        if count >= n_background:
            break
    background = torch.cat(bg_inputs, dim=0)[:n_background]

    # Get val encoder tensors for the target sample
    val_batches = []
    for batch in val_loader:
        x, _ = batch
        val_batches.append(x["encoder_cont"].to(device))
    val_enc = torch.cat(val_batches, dim=0)
    sample  = val_enc[sample_idx : sample_idx + 1]

    # Grab ref_x from val loader, trimmed to n_background
    # Using a temp var (_x) to avoid self-reference in the dict comprehension
    for batch in val_loader:
        _x, _ = batch
        ref_x = {
            k: v[:n_background].to(device) if isinstance(v, torch.Tensor) else v
            for k, v in _x.items()
        }
        break

    wrapper   = EncoderWrapper(model, ref_x)
    explainer = shap.GradientExplainer(wrapper, background)
    shap_vals = explainer.shap_values(sample)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]

    # Average |SHAP| over sequence length → (n_features,)
    shap_mean = np.abs(shap_vals[0]).mean(axis=0)
    shap_mean = shap_mean.squeze()  # remove extra dim if pred_len=1
    shap_series = (
        pd.Series(shap_mean, index=feature_names[:len(shap_mean)])
        .sort_values(ascending=False)
        .head(20)
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#d73027" if v > 0 else "#4575b4" for v in shap_series.values]
    ax.barh(shap_series.index[::-1], shap_series.values[::-1], color=colors[::-1])
    ax.set_title(
        f"SHAP Feature Attribution — Sample {sample_idx}\n"
        f"(mean |SHAP| over {TFT_CFG['max_encoder_length']}-day encoder window)"
    )
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
    df,
    train_dataset,
    ticker: str,
    checkpoint_path: str = None,
    save_dir: str = None,
) -> dict:
    from model import build_timeseries_dataset

    print(f"Explaining predictions for {ticker}...")
    model         = _load_model(checkpoint_path)
    feature_names = _get_feature_names(model)

    _, val_ds    = build_timeseries_dataset(df, cutoff=VAL_START)
    train_ds, _  = build_timeseries_dataset(df, cutoff=VAL_START)
    val_loader   = val_ds.to_dataloader(train=False, batch_size=64, num_workers=0)
    train_loader = train_ds.to_dataloader(train=True,  batch_size=64, num_workers=0)

    predictions  = model.predict(val_loader, mode="raw", return_x=True)
    enc_names    = feature_names["encoder"]
    static_names = feature_names["static_cats"] + feature_names["static_reals"]

    attn_save = os.path.join(save_dir, f"{ticker}_attention.png") if save_dir else None
    fig_attn  = attention_heatmap(model, predictions, enc_names,
                                  ticker=ticker, save_path=attn_save)

    imp_save = os.path.join(save_dir, f"{ticker}_importance.png") if save_dir else None
    fig_imp  = variable_importance(
        model, predictions, enc_names,
        static_feature_names=static_names or None,
        save_path=imp_save,
    )

    shap_save = os.path.join(save_dir, f"{ticker}_shap.png") if save_dir else None
    fig_shap  = shap_waterfall(
        model, train_loader, val_loader, enc_names,
        sample_idx=0, n_background=32, save_path=shap_save,
    )

    print(f"Done. Figures ready for {ticker}.")
    return {"attention": fig_attn, "importance": fig_imp, "shap": fig_shap}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MarketLens explainability")
    parser.add_argument("--ticker",     type=str, default="AAPL")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--save-dir",   type=str, default=None)
    parser.add_argument("--mock",       action="store_true")
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