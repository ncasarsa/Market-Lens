# AI Usage Disclosure
**Project:** MarketLens — Explainable Neural Trading Signal Platform
**Student:** Nathan Casarsa | CSCI 357 Spring 2026

---

## 1. What I Used AI For

**Architecture research and design**
Used Claude to research the Temporal Fusion Transformer architecture (Lim et al., 2021) and understand how PyTorch Forecasting's `TimeSeriesDataSet` handles heterogeneous input types (static categoricals, known future reals, unknown past reals). Claude explained the GroupNormalizer behavior and why `transformation=None` was necessary for log-return targets that include negative values.

**Debugging**
Used Claude extensively to debug:
- The `IndexError: index 52 is out of bounds` error caused by a feature count mismatch between the TFT checkpoint and the rebuilt val dataset after adding regime labels
- The `TypeError: '<' not supported between instances of 'int' and 'Timestamp'` error in `regime.py` caused by passing Timestamps to `np.searchsorted` which expects a homogeneous int64 array
- The `unknown operation None` error in `attention_heatmap()` caused by `interpret_output(reduction=None)` not being supported in the installed pytorch-forecasting version

**Code review and refactoring**
Used Claude to review `regime.py` and identify the bug where `label_clusters_by_rank` (which returns a dict) was being assigned inside a per-cluster loop, causing each `cluster_map[c]` to hold a dict instead of a label string. Also identified the duplicate KMeans run in `attach_regime_labels` that caused cluster ID misalignment.

**Writing boilerplate**
Used Claude to scaffold `app.py` Streamlit layout, CSS styling, and the demo notebook structure. The visual design (IBM Plex Mono, dark theme, regime color scheme) was iterated with Claude.

**Requirements and documentation**
Used Claude to generate `requirements.txt` and this `AI_USAGE.md`.

---

## 2. What I Did Not Use AI For

- **Problem formulation** — the decision to use TFT + LSTM autoencoder + FinBERT as a three-component architecture, and the choice to use regime labels as post-hoc overlays rather than TFT inputs, was my own design decision
- **Data pipeline design** — the structure of `build_dataset()` in `data.py`, including the cross-sectional relative return features (`sector_rel_return`, `market_rel_return`) and the FOMC countdown covariate, were designed independently
- **Rank-based regime labeling logic** — the insight that threshold-based labeling fails when all clusters have positive mean returns (as they did in a 2018–2024 bull market) and the decision to switch to rank-based assignment was my own diagnosis
- **Intra-window scoring fix** — the observation that scoring clusters on the single end-date forward return (5-day target) was decoupled from the window's actual price dynamics, and the fix to use mean intra-window daily log return instead, was my own analysis
- **Training runs** — all TFT and AE training was run independently in Colab with my own hyperparameter choices; the 84.74% direction accuracy result came from actual training, not from AI-generated code

---

## 3. How I Verified AI Output

- **Every code change was run end-to-end** in Colab before accepting it. Several Claude suggestions introduced new bugs (e.g., an early version of `attach_regime_labels` that still re-ran KMeans) which I caught by reading the output carefully and pushing back
- **The `label_clusters_by_rank` loop bug** was found by me reading the generated code and noticing that `cluster_map[c] = label_clusters_by_rank(cluster_stats)` assigned a whole dict to each key — Claude had provided the right function but the integration into the loop was wrong
- **Feature importance interpretation** — Claude suggested `market_rel_return` dominance was evidence of data leakage; I verified it was not by confirming the feature is computed from contemporaneous cross-sectional returns, not future data
- **The sentiment stub approach** — Claude's suggestion to zero-fill sentiment columns was verified by confirming the TFT checkpoint's `x_reals` list contained the sentiment column names, so the feature count would match

---

## 4. What I Learned from the Interaction

The most valuable thing I learned was how PyTorch Forecasting's `TimeSeriesDataSet` builds its internal `x_reals` list at training time and bakes that list into the Lightning checkpoint's `hyper_parameters`. Once a checkpoint is saved, the feature list is frozen — any mismatch between what the checkpoint expects and what the rebuilt dataset provides causes a silent index error rather than an informative one. Understanding this made the 52 vs 55 feature count bug immediately diagnosable.

I also learned the practical difference between scoring cluster quality on a single endpoint value (noisy, easily decoupled from the window's character) vs. the mean of intra-window values (directly reflects what the autoencoder actually saw). This is a general principle that applies beyond this project.
