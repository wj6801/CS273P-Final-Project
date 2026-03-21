import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
from ucimlrepo import fetch_ucirepo
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, matthews_corrcoef,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

RESULTS_DIR = "results"
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def load_data():
    parkinsons = fetch_ucirepo(id=174)
    df = parkinsons.data.original
    df["subject"] = df["name"].str.extract(r"(R\d+_S\d+)")[0]
    return df


def get_features(df):
    return [c for c in df.columns if c not in ("name", "status", "subject")]


class VoiceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_fold(df, train_idx, test_idx):
    features = get_features(df)
    X_tr = df.iloc[train_idx][features].values
    y_tr = df.iloc[train_idx]["status"].values
    X_te = df.iloc[test_idx][features].values
    y_te = df.iloc[test_idx]["status"].values
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    return X_tr, y_tr, X_te, y_te


class MLP(nn.Module):
    def __init__(self, input_dim=22, hidden_layers=[64, 32], dropout=0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_layers:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FocalLoss(nn.Module):
    """FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, n = 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    total_loss, n = 0, 0

    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits = model(X)
        total_loss += criterion(logits, y).item() * len(y)
        n += len(y)
        probs = torch.sigmoid(logits)
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend((probs > 0.5).float().cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    labels = np.array(all_labels)
    preds = np.array(all_preds)
    probs = np.array(all_probs)

    metrics = {
        "loss": total_loss / n,
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "mcc": matthews_corrcoef(labels, preds),
    }
    if len(np.unique(labels)) > 1:
        metrics["auc"] = roc_auc_score(labels, probs)
    else:
        metrics["auc"] = float("nan")
    return metrics


def train_model(model, train_loader, test_loader, criterion,
                lr=1e-3, n_epochs=200, patience=25, verbose=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_f1, best_metrics, wait = -1, None, 0
    history = {"train_loss": [], "test_loss": [], "test_f1": [], "test_auc": []}

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        test_metrics = evaluate(model, test_loader, criterion)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_metrics["loss"])
        history["test_f1"].append(test_metrics["f1"])
        history["test_auc"].append(test_metrics["auc"])

        if test_metrics["f1"] > best_f1:
            best_f1 = test_metrics["f1"]
            best_metrics = test_metrics
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"    Early stop at epoch {epoch}")
                break

        if verbose and epoch % 25 == 0:
            print(f"    Epoch {epoch:3d} | Train Loss: {train_loss:.4f} "
                  f"| Test F1: {test_metrics['f1']:.3f} AUC: {test_metrics['auc']:.3f}")

    return best_metrics, history


def run_cv(df, splits, model_fn, criterion, verbose=True):
    fold_metrics = []
    for i, (tr_idx, te_idx) in enumerate(splits):
        if verbose:
            print(f"  Fold {i+1}")
        X_tr, y_tr, X_te, y_te = prepare_fold(df, tr_idx, te_idx)
        train_loader = DataLoader(VoiceDataset(X_tr, y_tr), batch_size=32, shuffle=True)
        test_loader = DataLoader(VoiceDataset(X_te, y_te), batch_size=32, shuffle=False)
        model = model_fn().to(DEVICE)
        metrics, _ = train_model(model, train_loader, test_loader, criterion, verbose=verbose)
        fold_metrics.append(metrics)
        if verbose:
            print(f"    F1: {metrics['f1']:.3f}  AUC: {metrics['auc']:.3f}  MCC: {metrics['mcc']:.3f}")
    return fold_metrics


def summarize(fold_metrics, label=""):
    print(f"\n  {label}:")
    for k in ["accuracy", "f1", "auc", "mcc"]:
        vals = [m[k] for m in fold_metrics]
        print(f"    {k:12s}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")
    return {k: (float(np.mean([m[k] for m in fold_metrics])),
                float(np.std([m[k] for m in fold_metrics])))
            for k in ["accuracy", "f1", "auc", "mcc", "precision", "recall"]}


def experiment_baseline(df, splits, input_dim):
    print("\nBaseline MLP [64, 32] + Focal Loss (γ=2)")
    model_fn = lambda: MLP(input_dim, [64, 32], dropout=0.3)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    metrics = run_cv(df, splits, model_fn, criterion)
    return summarize(metrics, "Baseline")


def experiment_ablation_loss(df, splits, input_dim):
    print("\nLoss Function Ablation")
    configs = {
        "BCE": nn.BCEWithLogitsLoss(),
        "Weighted BCE": nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([48.0 / 147.0]).to(DEVICE)
        ),
        "Focal γ=1": FocalLoss(0.25, 1.0),
        "Focal γ=2": FocalLoss(0.25, 2.0),
        "Focal γ=3": FocalLoss(0.25, 3.0),
    }

    results = {}
    for name, criterion in configs.items():
        print(f"\n  {name}")
        model_fn = lambda: MLP(input_dim, [64, 32], dropout=0.3)
        metrics = run_cv(df, splits, model_fn, criterion, verbose=False)
        results[name] = summarize(metrics, name)

    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(results.keys())
    f1s = [results[n]["f1"][0] for n in names]
    stds = [results[n]["f1"][1] for n in names]
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974"]
    ax.bar(names, f1s, yerr=stds, capsize=5, color=colors)
    ax.set_ylabel("Mean F1 Score")
    ax.set_title("Loss Function Comparison (5-Fold CV)")
    ax.set_ylim(0.8, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "ablation_loss.png"), dpi=150)
    plt.close()

    return results


def experiment_ablation_architecture(df, splits, input_dim):
    print("\nArchitecture Ablation")
    configs = {
        "Shallow [32]": [32],
        "Medium [64, 32]": [64, 32],
        "Deep [128, 64, 32]": [128, 64, 32],
        "Wide [128, 128]": [128, 128],
        "Narrow-Deep [32, 32, 32]": [32, 32, 32],
    }

    results = {}
    for name, hidden in configs.items():
        print(f"\n  {name}")
        model_fn = lambda h=hidden: MLP(input_dim, h, dropout=0.3)
        criterion = FocalLoss(0.25, 2.0)
        metrics = run_cv(df, splits, model_fn, criterion, verbose=False)
        results[name] = summarize(metrics, name)

    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(results.keys())
    f1s = [results[n]["f1"][0] for n in names]
    stds = [results[n]["f1"][1] for n in names]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
    ax.bar(names, f1s, yerr=stds, capsize=5, color=colors)
    ax.set_ylabel("Mean F1 Score")
    ax.set_title("Architecture Comparison (5-Fold CV)")
    ax.set_ylim(0.8, 1.0)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "ablation_architecture.png"), dpi=150)
    plt.close()

    return results


def experiment_ablation_dropout(df, splits, input_dim):
    print("\nDropout Ablation")
    dropout_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = {}
    for d in dropout_vals:
        name = f"Dropout={d}"
        print(f"\n  {name}")
        model_fn = lambda dr=d: MLP(input_dim, [64, 32], dropout=dr)
        criterion = FocalLoss(0.25, 2.0)
        metrics = run_cv(df, splits, model_fn, criterion, verbose=False)
        results[name] = summarize(metrics, name)

    fig, ax = plt.subplots(figsize=(8, 4))
    f1s = [results[f"Dropout={d}"]["f1"][0] for d in dropout_vals]
    stds = [results[f"Dropout={d}"]["f1"][1] for d in dropout_vals]
    ax.errorbar(dropout_vals, f1s, yerr=stds, marker="o", capsize=5,
                linewidth=2, markersize=8, color="#2c3e50")
    ax.set_xlabel("Dropout Rate")
    ax.set_ylabel("Mean F1 Score")
    ax.set_title("Dropout Rate vs Performance (5-Fold CV)")
    ax.set_ylim(0.8, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "ablation_dropout.png"), dpi=150)
    plt.close()

    return results


def plot_training_curves(df, splits, input_dim, show=False):
    tr_idx, te_idx = splits[0]
    X_tr, y_tr, X_te, y_te = prepare_fold(df, tr_idx, te_idx)
    train_loader = DataLoader(VoiceDataset(X_tr, y_tr), batch_size=32, shuffle=True)
    test_loader = DataLoader(VoiceDataset(X_te, y_te), batch_size=32, shuffle=False)

    model = MLP(input_dim, [64, 32], dropout=0.3).to(DEVICE)
    criterion = FocalLoss(0.25, 2.0)
    _, history = train_model(model, train_loader, test_loader, criterion, verbose=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train", linewidth=2)
    axes[0].plot(history["test_loss"], label="Test", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Test Loss")
    axes[0].legend()

    axes[1].plot(history["test_f1"], label="F1", linewidth=2, color="#2ecc71")
    axes[1].plot(history["test_auc"], label="AUC", linewidth=2, color="#e74c3c")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Test F1 & AUC Over Training")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "training_curves.png"), dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    df = load_data()
    features = get_features(df)
    print(f"  {df.shape[0]} samples, {len(features)} features, {df['subject'].nunique()} subjects")
    print(f"  Class distribution: {dict(df['status'].value_counts())}")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(sgkf.split(df[features].values, df["status"].values, df["subject"].values))

    input_dim = len(features)
    baseline = experiment_baseline(df, splits, input_dim)
    loss_results = experiment_ablation_loss(df, splits, input_dim)
    arch_results = experiment_ablation_architecture(df, splits, input_dim)
    drop_results = experiment_ablation_dropout(df, splits, input_dim)
    plot_training_curves(df, splits, input_dim, show=False)

    all_results = {
        "baseline": baseline,
        "ablation_loss": loss_results,
        "ablation_architecture": arch_results,
        "ablation_dropout": drop_results,
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)