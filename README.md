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