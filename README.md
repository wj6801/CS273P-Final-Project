# Parkinson's Disease Detection from Voice Biomarkers

A custom PyTorch MLP for classifying Parkinson's disease from voice measurements, with Focal Loss for class imbalance and systematic ablation studies.

## Project Overview

This project classifies Parkinson's disease (PD) status from 22 voice biomarker features. We implement a custom MLP in PyTorch with Focal Loss and evaluate using subject-level cross-validation to prevent data leakage. Ablation studies compare loss functions, architectures, and dropout rates.

## Dataset

- **Source**: [UCI ML Repository – Parkinsons Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons) (fetched automatically via `ucimlrepo`)
- **Samples**: 195 recordings from 32 subjects
- **Features**: 22 voice measurements (jitter, shimmer, frequency, harmonics, nonlinear dynamics)
- **Target**: Binary (1 = Parkinson's, 0 = Healthy, 75:25 imbalance)

No manual data download is needed — the dataset is fetched at runtime using the `ucimlrepo` package. An internet connection is required on the first run.

## Setup

Tested with Python 3.12.

```bash
pip install -r requirements.txt
```

## How to Train & Evaluate

```bash
python main.py
```

This runs four experiments (~2 min on CPU):
1. Baseline MLP with Focal Loss
2. Loss function ablation (BCE, Weighted BCE, Focal γ=1,2,3)
3. Architecture ablation (shallow, medium, deep, wide, narrow-deep)
4. Dropout rate ablation (0.0 to 0.5)

Results are saved to `results/results.json` and figures to `results/figures/`.

For interactive exploration:
```bash
jupyter notebook demo.ipynb
```

The demo notebook reproduces the baseline 5-fold CV results (Table 1) and loss function ablation (Table 2) from the report.

## Expected Results

Baseline MLP [64, 32] + Focal Loss (5-fold subject-level CV):

| Metric   | Mean  | Std   |
|----------|-------|-------|
| Accuracy | 0.836 | 0.060 |
| F1       | 0.896 | 0.037 |
| AUC-ROC  | 0.750 | 0.181 |
| MCC      | 0.461 | 0.313 |

## Project Structure

```
├── main.py                    # All code (model, training, experiments)
├── demo.ipynb                 # Interactive demo notebook
├── results/
│   ├── results.json           # All metrics
│   └── figures/               # Plots
├── requirements.txt
└── README.md
```

## Dependencies

torch>=2.0, scikit-learn>=1.2, pandas>=1.5, matplotlib>=3.6, numpy<2, ucimlrepo
