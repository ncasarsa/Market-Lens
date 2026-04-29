"""
trainer.py
----------
Training loops for:
1. train_tft()          — main TFT training via PyTorch Lightning + W&B
2. train_regime_ae()    — LSTM autoencoder training for regime detection
3. run_sweep()          — W&B hyperparameter sweep launcher

Usage (from project root):
    python src/trainer.py --mode tft
    python src/trainer.py --mode regime
    python src/trainer.py --mode sweep
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import WandbLogger

import wandb

from config import (
    TFT as TFT_CFG,
    TRAIN as TRAIN_CFG,
    REGIME,
    WANDB as WANDB_CFG,
    SWEEP_CONFIG,
    CHECKPOINT_PATH,
    LOG_DIR,
    OHLCV_COLS,
    INDICATOR_COLS,
    VAL_START,
)


# ---------------------------------------------------------------------------
# 1. TFT Training
# ---------------------------------------------------------------------------

def train_tft(
    df: pd.DataFrame,
    cfg: dict = None,
    use_wandb: bool = True,
    run_name: str = None,
) -> tuple:
    """
    Trains the TFT model using PyTorch Lightning.

    Args:
        df:        cleaned DataFrame from data.py
        cfg:       hyperparameter overrides (defaults to config.TFT)
        use_wandb: whether to log to W&B
        run_name:  W&B run name (auto-generated if None)

    Returns:
        (trainer, model, train_dataset, val_dataset)
    """
    from model import build_timeseries_dataset, build_tft

    cfg = {**TFT_CFG, **(cfg or {})}

    # --- Datasets and dataloaders ---
    train_ds, val_ds = build_timeseries_dataset(
        df,
        max_encoder_length    = cfg["max_encoder_length"],
        max_prediction_length = cfg["max_prediction_length"],
        cutoff                = VAL_START,
    )

    train_loader = train_ds.to_dataloader(
        train      = True,
        batch_size = TRAIN_CFG["batch_size"],
        num_workers= TRAIN_CFG["num_workers"],
    )
    val_loader = val_ds.to_dataloader(
        train      = False,
        batch_size = TRAIN_CFG["batch_size"],
        num_workers= TRAIN_CFG["num_workers"],
    )

    # --- Model ---
    model = build_tft(train_ds, cfg)

    # --- Callbacks ---
    callbacks = [
        EarlyStopping(
            monitor  = "val_loss",
            patience = TRAIN_CFG["early_stopping_patience"],
            mode     = "min",
            verbose  = True,
        ),
        ModelCheckpoint(
            monitor   = "val_loss",
            dirpath   = str(CHECKPOINT_PATH.parent),
            filename  = "tft_best",
            save_top_k= 1,
            mode      = "min",
            verbose   = True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # --- Logger ---
    logger = None
    if use_wandb:
        logger = WandbLogger(
            project  = WANDB_CFG["project"],
            entity   = WANDB_CFG.get("entity"),
            name     = run_name,
            log_model= False,
        )
        logger.log_hyperparams(cfg)

    # --- Trainer ---
    trainer = Trainer(
        max_epochs         = TRAIN_CFG["max_epochs"],
        accelerator        = "auto",       # uses GPU if available, else CPU
        gradient_clip_val  = cfg["gradient_clip_val"],
        callbacks          = callbacks,
        logger             = logger,
        log_every_n_steps  = 10,
        enable_progress_bar= True,
        default_root_dir   = str(LOG_DIR),
    )

    print(f"\nStarting TFT training — {TRAIN_CFG['max_epochs']} max epochs, "
          f"early stopping patience={TRAIN_CFG['early_stopping_patience']}")
    trainer.fit(model, train_loader, val_loader)

    print("\nFinal Metrics:")
    for k, v in trainer.callback_metrics.items():
        print(f" {k}: {v:.6f}")

    best_val = trainer.callback_metrics.get("val_loss", None)
    print(f"\nTraining complete. Best val_loss: {best_val:.6f}" if best_val else
          "\nTraining complete.")

    if use_wandb:
        wandb.finish()

    return trainer, model, train_ds, val_ds


# ---------------------------------------------------------------------------
# 2. Regime Autoencoder Training
# ---------------------------------------------------------------------------

class RegimeLightningModule(pl.LightningModule):
    """
    Lightning wrapper around RegimeAutoencoder for clean training loop.
    """

    def __init__(self, model, learning_rate: float = 1e-3):
        super().__init__()
        self.model         = model
        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        recon = self(x)
        loss  = F.mse_loss(recon, x)
        self.log("train_recon_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        recon = self(x)
        loss  = F.mse_loss(recon, x)
        self.log("val_recon_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_recon_loss"},
        }


def train_regime_ae(
    df: pd.DataFrame,
    use_wandb: bool = True,
    run_name: str = "regime_ae",
) -> tuple:
    """
    Trains the LSTM autoencoder for market regime detection.

    Args:
        df:        cleaned DataFrame from data.py
        use_wandb: whether to log to W&B
        run_name:  W&B run name

    Returns:
        (trained RegimeAutoencoder, windows tensor, metadata list)
    """
    from model import RegimeAutoencoder, make_regime_windows

    feature_cols = OHLCV_COLS + INDICATOR_COLS

    # --- Windows ---
    windows, metadata = make_regime_windows(df, feature_cols)

    # Normalize each feature across all windows
    mean = windows.mean(dim=(0, 1), keepdim=True)
    std  = windows.std(dim=(0, 1), keepdim=True) + 1e-8
    windows_norm = (windows - mean) / std

    # Train / val split (80/20 by window index)
    n       = len(windows_norm)
    n_train = int(0.8 * n)
    train_ds = TensorDataset(windows_norm[:n_train])
    val_ds   = TensorDataset(windows_norm[n_train:])

    train_loader = DataLoader(
        train_ds,
        batch_size  = 256,
        shuffle     = True,
        num_workers = TRAIN_CFG["num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = 256,
        shuffle     = False,
        num_workers = TRAIN_CFG["num_workers"],
    )

    # --- Model ---
    ae_model = RegimeAutoencoder(input_size=len(feature_cols))
    lit_model = RegimeLightningModule(ae_model, learning_rate=1e-3)

    # --- Callbacks ---
    callbacks = [
        EarlyStopping(
            monitor  = "val_recon_loss",
            patience = 5,
            mode     = "min",
            verbose  = True,
        ),
        ModelCheckpoint(
            monitor   = "val_recon_loss",
            dirpath   = str(CHECKPOINT_PATH.parent),
            filename  = "regime_ae_best",
            save_top_k= 1,
            mode      = "min",
        ),
    ]

    # --- Logger ---
    logger = None
    if use_wandb:
        logger = WandbLogger(
            project  = WANDB_CFG["project"],
            entity   = WANDB_CFG.get("entity"),
            name     = run_name,
            log_model= False,
        )

    # --- Trainer ---
    trainer = Trainer(
        max_epochs          = 30,
        accelerator         = "auto",
        gradient_clip_val   = 1.0,
        callbacks           = callbacks,
        logger              = logger,
        log_every_n_steps   = 10,
        enable_progress_bar = True,
        default_root_dir    = str(LOG_DIR),
    )

    print(f"\nStarting Regime Autoencoder training...")
    trainer.fit(lit_model, train_loader, val_loader)

    if use_wandb:
        wandb.finish()

    return lit_model.model, windows, metadata


# ---------------------------------------------------------------------------
# 3. W&B Sweep
# ---------------------------------------------------------------------------

def run_sweep(df: pd.DataFrame, count: int = 20):
    """
    Launches a W&B Bayesian hyperparameter sweep over TFT config.

    Args:
        df:    cleaned DataFrame from data.py
        count: number of sweep runs
    """

    def sweep_run():
        run = wandb.init()
        sweep_cfg = dict(wandb.config)

        # Pull batch_size out separately (not a TFT param)
        batch_size = sweep_cfg.pop("batch_size", TRAIN_CFG["batch_size"])

        # Merge with defaults
        cfg = {**TFT_CFG, **sweep_cfg}

        from model import build_timeseries_dataset, build_tft

        train_ds, val_ds = build_timeseries_dataset(
            df,
            max_encoder_length    = cfg["max_encoder_length"],
            max_prediction_length = cfg["max_prediction_length"],
        )

        train_loader = train_ds.to_dataloader(
            train=True, batch_size=batch_size,
            num_workers=TRAIN_CFG["num_workers"],
        )
        val_loader = val_ds.to_dataloader(
            train=False, batch_size=batch_size,
            num_workers=TRAIN_CFG["num_workers"],
        )

        model = build_tft(train_ds, cfg)

        logger = WandbLogger(experiment=run)

        trainer = Trainer(
            max_epochs          = 20,           # shorter for sweeps
            accelerator         = "auto",
            gradient_clip_val   = cfg["gradient_clip_val"],
            callbacks           = [
                EarlyStopping(monitor="val_loss", patience=3, mode="min"),
            ],
            logger              = logger,
            log_every_n_steps   = 10,
            enable_progress_bar = False,        # cleaner sweep output
            default_root_dir    = str(LOG_DIR),
        )

        trainer.fit(model, train_loader, val_loader)
        wandb.finish()

    sweep_id = wandb.sweep(
        SWEEP_CONFIG,
        project = WANDB_CFG["project"],
        entity  = WANDB_CFG.get("entity"),
    )
    print(f"Sweep ID: {sweep_id}")
    wandb.agent(sweep_id, function=sweep_run, count=count)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MarketLens trainer")
    parser.add_argument(
        "--mode", choices=["tft", "regime", "sweep"], default="tft",
        help="What to train: tft | regime | sweep"
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable W&B logging"
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock data (no network required)"
    )
    parser.add_argument(
        "--sweep-count", type=int, default=20,
        help="Number of sweep runs (sweep mode only)"
    )
    args = parser.parse_args()

    # Load data
    from data import (
        build_dataset, _make_mock_ohlcv,
        add_technical_indicators, add_target,
        add_time_idx, clean_and_normalize,
    )
    from config import DATASET_PATH, TICKERS, TRAIN_START, TRAIN_END

    if args.mock:
        print("Using mock data...")
        tickers = ["AAPL", "MSFT", "XOM", "JPM", "NEE",
                   "NVDA", "BAC", "LMT", "NEE", "CVX"]
        df = _make_mock_ohlcv(tickers, "2020-01-01", "2023-12-31")
        df = add_technical_indicators(df)
        df = add_target(df)
        df = add_time_idx(df)
        df = clean_and_normalize(df)
    elif DATASET_PATH.exists():
        print(f"Loading cached dataset from {DATASET_PATH}...")
        df = pd.read_parquet(DATASET_PATH)
    else:
        print("Downloading full dataset...")
        df = build_dataset(
            tickers   = TICKERS,
            start     = TRAIN_START,
            end       = TRAIN_END,
            save_path = str(DATASET_PATH),
        )

    use_wandb = not args.no_wandb

    if args.mode == "tft":
        train_tft(df, use_wandb=use_wandb)

    elif args.mode == "regime":
        train_regime_ae(df, use_wandb=use_wandb)

    elif args.mode == "sweep":
        if not use_wandb:
            print("ERROR: sweeps require W&B. Remove --no-wandb.")
            sys.exit(1)
        run_sweep(df, count=args.sweep_count)