# Reliability-Aware Graph Conformal Prediction — Reproduction Guide

This submission contains the code for team 41 project on improving conformal
prediction (CP) for graph neural networks. Starting from **DAPS** (Diffusion
Adaptive Prediction Sets), we study two stages of the conformal prediction pipeline:

**Calibration stage**

- **Calibration split by centrality** — the calibration set is partitioned into
  buckets by node centrality, and a separate quantile q̂ is computed per
  bucket.

**Score stage** — two ways of modifying the score function itself:

- **CM-DAPS** — a centrality-weighted score function, where graph structure
  (centrality) determines how much each neighbor contributes to score
  diffusion.
- **GSC-PPR / DC-GSC-PPR** — a model-based reliability approach, where the
  model's own confidence and local agreement guide score diffusion, with an
  optional degree-stratified calibration step.

## Submitted file

The submission consists of a single notebook, `code.ipynb`, which is a
`.ipynb` file intended to be run on **Google Colab (free version)**.

Running it reproduces all of our results:

- the relationship between node centrality and model error rate
- the comparison plots including APS, DAPS, CM-DAPS, 2-hop DAPS, and 2-hop CM-DAPS
- the calibration split by centrality
- the lambda (λ) trade-off analysis for CM-DAPS
- the full GSC-PPR and DC-GSC-PPR experiments, including the degree-bin
  coverage analysis and the grand sweep.

## GPU usage

**We used a GPU.** The notebook was run on the Colab free-tier GPU
runtime (NVIDIA T4). It automatically selects the device:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

If no GPU is available the code falls back to CPU and still runs correctly,
but training and the grand sweep will be noticeably slower. To reproduce our
results as submitted, please enable the GPU runtime:

> **Runtime → Change runtime type → GPU**

## Datasets

All datasets are **downloaded automatically
by the code on first run** into a local `./data` directory — no dataset files
need to be opened or placed manually:

- **Amazon Photo / Amazon Computers** — via `torch_geometric.datasets.Amazon`
- **CoraML / PubMed / CiteSeer** — via `torch_geometric.datasets.Planetoid`


## How to reproduce

1. Upload `team_41_code_submission.ipynb` to **Google Colab** (free version).
2. Enable the GPU runtime: **Runtime → Change runtime type → GPU**.
3. Run all cells from top to bottom: **Runtime → Run all**.
   - The first cell installs the non-default package with
     `!pip install -q torch_geometric`. All other packages (`torch`, `numpy`,
     `pandas`, `matplotlib`, `scipy`, `networkx`) are pre-installed on Colab,
     so no extra installation is needed.
   - The dataset-loading cells download the data automatically.

A fixed random seed is set near the top of the notebook for reproducibility.
Minor numerical differences may still occur due to library versions or GPU
vs. CPU execution, but the reported trends are stable.

## Notes on runtime

- The grand sweep evaluates 5 datasets × 3 models × 4 methods × 5 calibration
  seeds and is the most time-consuming part. On the Colab free-tier GPU it
  completes within a normal session; on CPU it is significantly slower.
- Run the notebook in a single session from top to bottom so that all
  intermediate variables remain defined.
