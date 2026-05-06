"""
app.py
------
MarketLens Streamlit dashboard.

Tabs:
    1. Predictions  — price chart + predicted return with quantile bands + regime overlay
    2. Explainability — feature importance + attention heatmap + SHAP waterfall
    3. Regime Analysis — PCA scatter + regime timeline + cluster stats

Run in Colab:
    !streamlit run /content/Market-Lens/app.py --server.port 8501
    # then expose via ngrok
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title   = "MarketLens",
    page_icon    = "📡",
    layout       = "wide",
    initial_sidebar_state = "expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d0f14;
    color: #e2e8f0;
}

h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: -0.02em;
}

.stTabs [data-baseweb="tab-list"] {
    background-color: #161922;
    border-bottom: 1px solid #2d3748;
    gap: 0;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #718096;
    padding: 10px 24px;
    border-bottom: 2px solid transparent;
}

.stTabs [aria-selected="true"] {
    color: #63b3ed !important;
    border-bottom: 2px solid #63b3ed !important;
    background: transparent !important;
}

.metric-card {
    background: #161922;
    border: 1px solid #2d3748;
    border-radius: 6px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #718096;
    margin-bottom: 4px;
}

.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #e2e8f0;
}

.regime-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 3px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    font-weight: 600;
}

.bull     { background: #1a3a2a; color: #68d391; border: 1px solid #2f855a; }
.bear     { background: #3a1a1a; color: #fc8181; border: 1px solid #c53030; }
.volatile { background: #3a2a1a; color: #f6ad55; border: 1px solid #c05621; }
.sideways { background: #1a2a3a; color: #63b3ed; border: 1px solid #2b6cb0; }
.unknown  { background: #2d3748; color: #718096; border: 1px solid #4a5568; }

.sidebar-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4a5568;
    margin-bottom: 8px;
    margin-top: 20px;
}

[data-testid="stSidebar"] {
    background-color: #0d0f14;
    border-right: 1px solid #1a2030;
}

.stSelectbox > div > div {
    background-color: #161922;
    border: 1px solid #2d3748;
    color: #e2e8f0;
}

.stSlider > div { color: #e2e8f0; }

hr { border-color: #2d3748; }

.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4a5568;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1a2030;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REGIME_COLORS = {
    "bull":     "#68d391",
    "bear":     "#fc8181",
    "volatile": "#f6ad55",
    "sideways": "#63b3ed",
    "unknown":  "#718096",
}

DRIVE_BASE   = "/content/drive/MyDrive/MarketLens_checkpoints"
TFT_CKPT     = f"{DRIVE_BASE}/tft_best_5day_baseline.ckpt"
AE_CKPT      = f"{DRIVE_BASE}/regime_ae_best_15t.ckpt"
DATASET_PATH = f"{DRIVE_BASE}/dataset_15t.parquet"


# ---------------------------------------------------------------------------
# Data / model loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    df = pd.read_parquet(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    # Stub sentiment columns if not present — TFT checkpoint expects them
    for col in ["sentiment_score", "sentiment_volume", "sentiment_rolling_3d"]:
        if col not in df.columns:
            df[col] = 0.0
    return df


@st.cache_resource(show_spinner="Loading TFT checkpoint...")
def load_tft():
    from model import MarketTFT, build_timeseries_dataset
    from config import VAL_START, TRAIN as TRAIN_CFG

    df = load_data()
    train_ds, val_ds = build_timeseries_dataset(df, cutoff=VAL_START)

    model = MarketTFT.load_from_checkpoint(TFT_CKPT)
    model.eval()

    val_loader = val_ds.to_dataloader(
        train=False, batch_size=128, num_workers=0
    )
    train_loader = train_ds.to_dataloader(
        train=False, batch_size=128, num_workers=0
    )
    return model, train_ds, val_ds, train_loader, val_loader


@st.cache_resource(show_spinner="Loading Regime Autoencoder...")
def load_regime_ae():
    from model import RegimeAutoencoder, make_regime_windows
    from config import OHLCV_COLS, INDICATOR_COLS

    df = load_data()
    feature_cols = OHLCV_COLS + INDICATOR_COLS

    ckpt = torch.load(AE_CKPT, map_location="cpu")
    if "state_dict" in ckpt:
        ae = RegimeAutoencoder(input_size=len(feature_cols))
        state = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
        ae.load_state_dict(state)
    else:
        ae = ckpt

    ae.eval()
    windows, metadata = make_regime_windows(df, feature_cols)
    return ae, windows, metadata, feature_cols


@st.cache_data(show_spinner="Running predictions...")
def get_predictions():
    model, train_ds, val_ds, train_loader, val_loader = load_tft()

    with torch.no_grad():
        predictions = model.predict(val_loader, mode="raw", return_x=True)

    pred_all    = predictions.output.prediction.squeeze(1).cpu()  # (N, 7)
    pred_median = pred_all[:, 3]
    actuals     = torch.cat([y[0] for x, y in val_loader]).squeeze(-1).cpu()

    mae     = (pred_median - actuals).abs().mean().item()
    rmse    = ((pred_median - actuals) ** 2).mean().sqrt().item()
    dir_acc = ((pred_median > 0) == (actuals > 0)).float().mean().item()

    return predictions, pred_all, pred_median, actuals, mae, rmse, dir_acc


@st.cache_data(show_spinner="Clustering regimes...")
def get_regime_labels():
    from regime import fit_regimes, attach_regime_labels
    from config import REGIME

    df       = load_data()
    ae, windows, metadata, feature_cols = load_regime_ae()

    # fit_regimes encodes, clusters, labels, and saves regime_labels.parquet
    cluster_map = fit_regimes(ae, windows, metadata, df,
                              n_clusters=REGIME["n_clusters"])

    # attach reads the saved parquet — no second KMeans run
    df_labeled = attach_regime_labels(df, cluster_map)

    return df_labeled, cluster_map, windows, metadata


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("# 📡 MarketLens")
    st.markdown(
        '<div style="font-family:IBM Plex Mono;font-size:0.7rem;color:#4a5568;margin-top:-8px;">'
        'CSCI 357 · Nathan Casarsa</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    try:
        from config import TICKERS, VAL_START, TEST_START
        df_side = load_data()
        tickers_available = sorted(df_side["ticker"].unique().tolist())
    except Exception:
        tickers_available = ["AAPL", "MSFT", "NVDA"]
        VAL_START  = "2022-01-01"
        TEST_START = "2024-01-01"

    st.markdown('<div class="sidebar-header">Ticker</div>', unsafe_allow_html=True)
    selected_ticker = st.selectbox("", tickers_available, label_visibility="collapsed")

    st.markdown('<div class="sidebar-header">Date Range</div>', unsafe_allow_html=True)
    date_range = st.select_slider(
        "",
        options=["2018", "2019", "2020", "2021", "2022", "2023", "2024"],
        value=("2022", "2024"),
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-header">Val / Test Split</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-family:IBM Plex Mono;font-size:0.7rem;color:#718096;">'
        f'Val  → {VAL_START}<br>Test → {TEST_START}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown('<div class="sidebar-header">Models</div>', unsafe_allow_html=True)

    models_loaded = os.path.exists(TFT_CKPT) and os.path.exists(AE_CKPT)
    status_color  = "#68d391" if models_loaded else "#fc8181"
    status_text   = "READY"   if models_loaded else "NOT FOUND"
    st.markdown(
        f'<div style="font-family:IBM Plex Mono;font-size:0.7rem;">'
        f'TFT  <span style="color:{status_color}">● {status_text}</span><br>'
        f'AE   <span style="color:{status_color}">● {status_text}</span></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "01 · PREDICTIONS",
    "02 · EXPLAINABILITY",
    "03 · REGIME ANALYSIS",
])


# ===========================================================================
# TAB 1 — Predictions
# ===========================================================================
with tab1:
    st.markdown('<div class="section-label">Next-Day Log Return Forecast</div>', unsafe_allow_html=True)

    try:
        predictions, pred_all, pred_median, actuals, mae, rmse, dir_acc = get_predictions()
        df_labeled, cluster_map, windows, metadata = get_regime_labels()

        # --- Metrics row ---
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Val MAE</div>'
                f'<div class="metric-value">{mae:.4f}</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Val RMSE</div>'
                f'<div class="metric-value">{rmse:.4f}</div></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Direction Acc</div>'
                f'<div class="metric-value">{dir_acc:.1%}</div></div>',
                unsafe_allow_html=True,
            )
        with c4:
            ticker_rows   = df_labeled[df_labeled["ticker"] == selected_ticker]
            ticker_regime = ticker_rows["regime"].iloc[-1] if len(ticker_rows) else "unknown"
            regime_class  = ticker_regime.lower()
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Current Regime</div>'
                f'<div style="margin-top:8px">'
                f'<span class="regime-pill {regime_class}">{ticker_regime.upper()}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        # --- Price + Regime overlay chart ---
        ticker_df = df_labeled[
            (df_labeled["ticker"] == selected_ticker) &
            (df_labeled["date"] >= pd.Timestamp(f"{date_range[0]}-01-01")) &
            (df_labeled["date"] <= pd.Timestamp(f"{date_range[1]}-12-31"))
        ].copy().sort_values("date").reset_index(drop=True)

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 7),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )
        fig.patch.set_facecolor("#0d0f14")
        for ax in [ax1, ax2]:
            ax.set_facecolor("#0d0f14")
            ax.tick_params(colors="#718096", labelsize=8)
            ax.spines[:].set_color("#2d3748")

        ax1.plot(ticker_df["date"], ticker_df["close"],
                 color="#63b3ed", linewidth=1.2, alpha=0.9)
        ax1.set_ylabel("Close Price", color="#718096", fontsize=9, fontfamily="monospace")
        ax1.set_title(f"{selected_ticker} · Price & Regime",
                      color="#e2e8f0", fontsize=11, fontfamily="monospace", pad=12)

        # Regime color bands on price chart
        for i in range(len(ticker_df) - 1):
            regime = ticker_df["regime"].iloc[i]
            color  = REGIME_COLORS.get(regime, "#718096")
            ax1.axvspan(ticker_df["date"].iloc[i], ticker_df["date"].iloc[i + 1],
                        alpha=0.12, color=color, linewidth=0)

        # Regime scatter strip
        for regime, color in REGIME_COLORS.items():
            mask = ticker_df["regime"] == regime
            if mask.any():
                ax2.scatter(ticker_df.loc[mask, "date"], [regime] * mask.sum(),
                            c=color, s=8, alpha=0.8, linewidths=0)

        ax2.set_ylabel("Regime", color="#718096", fontsize=9, fontfamily="monospace")
        ax2.set_yticks(list(REGIME_COLORS.keys()))
        ax2.set_yticklabels(list(REGIME_COLORS.keys()),
                             color="#718096", fontsize=7, fontfamily="monospace")

        patches = [
            mpatches.Patch(color=c, label=r.capitalize(), alpha=0.7)
            for r, c in REGIME_COLORS.items() if r != "unknown"
        ]
        ax1.legend(handles=patches, loc="upper left", fontsize=7,
                   facecolor="#161922", edgecolor="#2d3748", labelcolor="#e2e8f0")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # --- Prediction vs Actual chart ---
        st.markdown(
            '<div class="section-label" style="margin-top:24px;">'
            'Val Set: Predicted vs Actual (z-score normalized)</div>',
            unsafe_allow_html=True,
        )

        n_plot = min(300, len(pred_median))
        fig2, ax = plt.subplots(figsize=(14, 4))
        fig2.patch.set_facecolor("#0d0f14")
        ax.set_facecolor("#0d0f14")
        ax.tick_params(colors="#718096", labelsize=8)
        ax.spines[:].set_color("#2d3748")

        x = range(n_plot)
        ax.fill_between(x, pred_all[:n_plot, 0].numpy(), pred_all[:n_plot, 6].numpy(),
                        alpha=0.10, color="#63b3ed", label="2%–98% interval")
        ax.fill_between(x, pred_all[:n_plot, 1].numpy(), pred_all[:n_plot, 5].numpy(),
                        alpha=0.15, color="#63b3ed", label="10%–90% interval")
        ax.fill_between(x, pred_all[:n_plot, 2].numpy(), pred_all[:n_plot, 4].numpy(),
                        alpha=0.20, color="#63b3ed", label="25%–75% interval")
        ax.plot(actuals[:n_plot].numpy(),
                color="#e2e8f0", linewidth=0.8, alpha=0.6, label="Actual")
        ax.plot(pred_median[:n_plot].numpy(),
                color="#63b3ed", linewidth=1.2, label="Predicted (median)")
        ax.axhline(0, color="#4a5568", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Sample index", color="#718096", fontsize=9, fontfamily="monospace")
        ax.set_ylabel("Normalized return", color="#718096", fontsize=9, fontfamily="monospace")
        ax.legend(fontsize=7, facecolor="#161922", edgecolor="#2d3748", labelcolor="#e2e8f0")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    except Exception as e:
        st.error(f"Could not load predictions: {e}")
        st.info("Make sure TFT checkpoint exists at the Drive path and the dataset parquet is loaded.")


# ===========================================================================
# TAB 2 — Explainability
# ===========================================================================
with tab2:
    st.markdown('<div class="section-label">Model Interpretability</div>', unsafe_allow_html=True)

    try:
        from explainability import attention_heatmap, variable_importance, shap_waterfall
        model, train_ds, val_ds, train_loader, val_loader = load_tft()

        with torch.no_grad():
            predictions_exp = model.predict(val_loader, mode="raw", return_x=True)

        col_imp, col_attn = st.columns(2)

        # --- Variable Importance ---
        with col_imp:
            st.markdown('<div class="section-label">Feature Importance</div>', unsafe_allow_html=True)
            try:
                interpretation = model.interpret_output(
                    predictions_exp.output, reduction="mean"
                )
                enc_imp   = interpretation["encoder_variables"].cpu().numpy()
                enc_names = train_ds.reals if hasattr(train_ds, "reals") else [
                    f"feat_{i}" for i in range(len(enc_imp))
                ]
                imp_series = (
                    pd.Series(enc_imp, index=enc_names[:len(enc_imp)])
                    .sort_values(ascending=True)
                    .tail(20)
                )

                fig3, ax = plt.subplots(figsize=(7, 8))
                fig3.patch.set_facecolor("#0d0f14")
                ax.set_facecolor("#0d0f14")
                ax.tick_params(colors="#718096", labelsize=8)
                ax.spines[:].set_color("#2d3748")

                colors_imp = [
                    "#63b3ed" if v < imp_series.median() else "#90cdf4"
                    for v in imp_series.values
                ]
                ax.barh(imp_series.index, imp_series.values, color=colors_imp, alpha=0.85)
                ax.set_title("Encoder Variable Selection",
                             color="#e2e8f0", fontsize=10, fontfamily="monospace")
                ax.set_xlabel("Attention weight", color="#718096",
                              fontsize=8, fontfamily="monospace")
                plt.tight_layout()
                st.pyplot(fig3)
                plt.close()
            except Exception as e:
                st.warning(f"Variable importance error: {e}")

        # --- Attention Heatmap ---
        with col_attn:
            st.markdown('<div class="section-label">Temporal Attention</div>', unsafe_allow_html=True)
            try:
                fig4 = attention_heatmap(
                    model, predictions_exp,
                    feature_names=[],
                    ticker=selected_ticker,
                    n_samples=30,
                )
                fig4.patch.set_facecolor("#0d0f14")
                st.pyplot(fig4)
                plt.close()
            except Exception as e:
                st.warning(f"Attention heatmap error: {e}")

        # --- SHAP ---
        st.markdown(
            '<div class="section-label" style="margin-top:24px;">SHAP Feature Attribution</div>',
            unsafe_allow_html=True,
        )
        shap_sample = st.slider("Sample index to explain", 0, 200, 0)

        if st.button("Run SHAP (takes ~2 min)", type="primary"):
            with st.spinner("Computing SHAP values..."):
                try:
                    enc_names_shap = train_ds.reals if hasattr(train_ds, "reals") else []
                    fig5 = shap_waterfall(
                        model, train_loader, val_loader,
                        feature_names=enc_names_shap,
                        sample_idx=shap_sample,
                        n_background=50,
                    )
                    fig5.patch.set_facecolor("#0d0f14")
                    st.pyplot(fig5)
                    plt.close()
                except Exception as e:
                    st.error(f"SHAP error: {e}")

    except Exception as e:
        st.error(f"Could not load explainability module: {e}")


# ===========================================================================
# TAB 3 — Regime Analysis
# ===========================================================================
with tab3:
    st.markdown('<div class="section-label">Unsupervised Market Regime Detection</div>', unsafe_allow_html=True)

    try:
        df_labeled, cluster_map, windows, metadata = get_regime_labels()
        ae, _, _, feature_cols = load_regime_ae()

        # --- Cluster stats ---
        st.markdown('<div class="section-label">Cluster Statistics</div>', unsafe_allow_html=True)
        stat_cols     = st.columns(4)
        regime_counts = df_labeled["regime"].value_counts()

        for i, (regime, color) in enumerate(
            [(r, c) for r, c in REGIME_COLORS.items() if r != "unknown"]
        ):
            with stat_cols[i]:
                count = regime_counts.get(regime, 0)
                pct   = count / len(df_labeled) * 100
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">'
                    f'<span class="regime-pill {regime}">{regime.upper()}</span></div>'
                    f'<div class="metric-value" style="color:{color}">{count:,}</div>'
                    f'<div style="font-family:IBM Plex Mono;font-size:0.7rem;color:#718096;">'
                    f'{pct:.1f}% of rows</div></div>',
                    unsafe_allow_html=True,
                )

        col_pca, col_timeline = st.columns([1, 1])

        # --- PCA scatter — uses regime labels from saved parquet, no second KMeans ---
        with col_pca:
            st.markdown('<div class="section-label">Latent Space (PCA 2D)</div>', unsafe_allow_html=True)
            try:
                from config import DATA_DIR
                regime_df = pd.read_parquet(DATA_DIR / "regime_labels.parquet")

                mean_w   = windows.mean(dim=(0, 1), keepdim=True)
                std_w    = windows.std(dim=(0, 1),  keepdim=True) + 1e-8
                win_norm = (windows - mean_w) / std_w

                with torch.no_grad():
                    latents = ae.encode(win_norm).numpy()

                scaler         = StandardScaler()
                latents_scaled = scaler.fit_transform(latents)

                pca        = PCA(n_components=2)
                latents_2d = pca.fit_transform(latents_scaled)

                # Use the saved regime labels — aligned 1:1 with windows/metadata
                regime_labels_arr = regime_df["regime"].values

                fig6, ax = plt.subplots(figsize=(7, 6))
                fig6.patch.set_facecolor("#0d0f14")
                ax.set_facecolor("#0d0f14")
                ax.tick_params(colors="#718096", labelsize=7)
                ax.spines[:].set_color("#2d3748")

                for regime_label, color in REGIME_COLORS.items():
                    if regime_label == "unknown":
                        continue
                    mask = regime_labels_arr == regime_label
                    if mask.any():
                        ax.scatter(
                            latents_2d[mask, 0], latents_2d[mask, 1],
                            c=color, s=4, alpha=0.35,
                            label=regime_label.capitalize(), linewidths=0,
                        )

                ax.set_title("Regime Latent Space",
                             color="#e2e8f0", fontsize=10, fontfamily="monospace")
                ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                              color="#718096", fontsize=8, fontfamily="monospace")
                ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
                              color="#718096", fontsize=8, fontfamily="monospace")
                ax.legend(fontsize=7, facecolor="#161922", edgecolor="#2d3748",
                          labelcolor="#e2e8f0", markerscale=3)
                plt.tight_layout()
                st.pyplot(fig6)
                plt.close()

            except Exception as e:
                st.warning(f"PCA scatter error: {e}")

        # --- Regime timeline ---
        with col_timeline:
            st.markdown(
                f'<div class="section-label">Regime Timeline — {selected_ticker}</div>',
                unsafe_allow_html=True,
            )
            try:
                ticker_df_r = df_labeled[
                    (df_labeled["ticker"] == selected_ticker) &
                    (df_labeled["date"] >= pd.Timestamp(f"{date_range[0]}-01-01")) &
                    (df_labeled["date"] <= pd.Timestamp(f"{date_range[1]}-12-31"))
                ].sort_values("date").reset_index(drop=True)

                fig7, (ax1, ax2) = plt.subplots(
                    2, 1, figsize=(7, 6),
                    gridspec_kw={"height_ratios": [2, 1]},
                    sharex=True,
                )
                fig7.patch.set_facecolor("#0d0f14")
                for ax in [ax1, ax2]:
                    ax.set_facecolor("#0d0f14")
                    ax.tick_params(colors="#718096", labelsize=7)
                    ax.spines[:].set_color("#2d3748")

                ax1.plot(ticker_df_r["date"], ticker_df_r["close"],
                         color="#63b3ed", linewidth=1.0, alpha=0.9)
                ax1.set_ylabel("Close", color="#718096", fontsize=8, fontfamily="monospace")
                ax1.set_title(f"{selected_ticker} · Regime Timeline",
                              color="#e2e8f0", fontsize=10, fontfamily="monospace")

                for regime, color in REGIME_COLORS.items():
                    mask = ticker_df_r["regime"] == regime
                    if mask.any():
                        ax2.scatter(
                            ticker_df_r.loc[mask, "date"],
                            [regime] * mask.sum(),
                            c=color, s=6, alpha=0.9, linewidths=0,
                        )

                ax2.set_yticks(list(REGIME_COLORS.keys()))
                ax2.set_yticklabels(list(REGIME_COLORS.keys()),
                                     color="#718096", fontsize=6, fontfamily="monospace")
                ax2.set_ylabel("Regime", color="#718096", fontsize=8, fontfamily="monospace")

                plt.tight_layout()
                st.pyplot(fig7)
                plt.close()

            except Exception as e:
                st.warning(f"Timeline error: {e}")

        # --- Per-regime return stats ---
        st.markdown(
            f'<div class="section-label" style="margin-top:24px;">'
            f'Return Stats by Regime — {selected_ticker}</div>',
            unsafe_allow_html=True,
        )
        try:
            ticker_regime_df = df_labeled[df_labeled["ticker"] == selected_ticker]
            stats = (
                ticker_regime_df
                .groupby("regime")["target"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            stats.columns = ["Regime", "Mean Return", "Std Dev", "Count"]
            stats["Mean Return"] = stats["Mean Return"].map("{:.5f}".format)
            stats["Std Dev"]     = stats["Std Dev"].map("{:.5f}".format)
            st.dataframe(stats, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Stats table error: {e}")

    except Exception as e:
        st.error(f"Could not load regime analysis: {e}")
        st.info("Make sure the Regime AE checkpoint exists and the dataset is loaded.")