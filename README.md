# MarketLens

An explainable neural trading signal platform built with a Temporal Fusion Transformer (TFT),
FinBERT sentiment encoding, and LSTM-based regime detection. Deployed as an interactive
Streamlit dashboard on Hugging Face Spaces.

> CSCI 357 Final Project — Spring 2026

## Live Demo

[Hugging Face Spaces](https://huggingface.co/spaces/YOUR_USERNAME/marketlens) *(coming soon)*

---

## Architecture

| Component | Description |
|---|---|
| TFT | Core multi-horizon forecasting model |
| FinBERT | Financial news sentiment encoder |
| LSTM Autoencoder | Unsupervised market regime detection |
| DeepSHAP | Per-prediction feature attribution |
| Streamlit | Interactive dashboard |

---

## Project Structure

Market-Lens/
├── src/
│   ├── data.py              ✅ done
│   ├── model.py             ✅ done
│   ├── trainer.py           ✅ done
│   ├── explainability.py    ✅ done
│   ├── regime.py            Needs to be tested
│   └── sentiment.py         Needs fixing to parse through externally built dataset
├── app.py                   ✅ done
├── sweep.py                 ❌ not started
├── config.py                ✅ done
├── data/
│   └── dataset.parquet      ✅ generated in Colab
├── models/
│   ├── tft_best.ckpt        ✅ generated in Colab
│   └── regime_ae_best.ckpt  ✅ generated in Colab
├── logs/                    ✅ auto-created
├── notebooks/
│   └── MarketLens_Training.ipynb  ✅ done
├── requirements.txt          ✅ done
├── .gitignore               ✅ done
└── README.md                ✅ done
