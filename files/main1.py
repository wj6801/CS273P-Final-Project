import copy
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from ucimlrepo import fetch_ucirepo
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_data():
    parkinsons = fetch_ucirepo(id=174)
    df = parkinsons.data.original.copy()
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


class MLP(nn.Module):
    def __init__(self, input_dim, hidden=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden=64, dropout=0.3):
        super().__init__()
        self.in_layer = nn.Linear(input_dim, hidden)
        self.block1 = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.block2 = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.in_layer(x)
        h = h + self.block1(h)
        h = h + self.block2(h)
        return self.out(h).squeeze(-1)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class HybridLoss(nn.Module):
    def __init__(self, pos_weight, alpha=0.25, gamma=2.0, lam=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.lam = lam

    def forward(self, logits, targets):
        return self.lam * self.bce(logits, targets) + (1 - self.lam) * self.focal(logits, targets)


def make_model(input_dim, model_type):
    if model_type == "residual":
        return ResidualMLP(input_dim=input_dim, hidden=64, dropout=0.3).to(DEVICE)
    return MLP(input_dim=input_dim, hidden=64, dropout=0.3).to(DEVICE)


def predict_probs(model, loader):
    model.eval()
    probs = []
    ys = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            p = torch.sigmoid(logits).cpu().numpy()
            probs.append(p)
            ys.append(yb.numpy())
    probs = np.concatenate(probs)
    ys = np.concatenate(ys)
    return probs, ys


def train_one_fold(X_train, y_train, X_val, y_val, model_type, loss_type, use_scheduler, tune_threshold):
    train_loader = DataLoader(VoiceDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(VoiceDataset(X_val, y_val), batch_size=128, shuffle=False)

    model = make_model(input_dim=X_train.shape[1], model_type=model_type)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=4
        )

    pos = max(1, int((y_train == 1).sum()))
    neg = max(1, int((y_train == 0).sum()))
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=DEVICE)

    if loss_type == "focal":
        criterion = FocalLoss(alpha=0.35, gamma=2.0)
    elif loss_type == "hybrid":
        criterion = HybridLoss(pos_weight=pos_weight, alpha=0.35, gamma=2.0, lam=0.5)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    wait = 0
    patience = 12
    epochs = 80

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        val_probs, val_true = predict_probs(model, val_loader)
        if len(np.unique(val_true)) < 2:
            val_auc = 0.5
        else:
            val_auc = roc_auc_score(val_true, val_probs)

        if scheduler is not None:
            scheduler.step(val_auc)

        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    val_probs, val_true = predict_probs(model, val_loader)

    threshold = 0.5
    if tune_threshold:
        best_f1 = -1.0
        for t in np.linspace(0.2, 0.8, 31):
            val_pred = (val_probs >= t).astype(int)
            score = f1_score(val_true, val_pred)
            if score > best_f1:
                best_f1 = score
                threshold = float(t)

    return model, threshold


def evaluate_experiment(df, exp):
    features = get_features(df)
    X = df[features].values
    y = df["status"].values.astype(int)
    groups = df["subject"].values

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_metrics = []

    print(f"\n=== {exp['name']} ===")
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups), start=1):
        X_tr_full = X[train_idx]
        y_tr_full = y[train_idx]
        g_tr_full = groups[train_idx]
        X_te = X[test_idx]
        y_te = y[test_idx]

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        tr_rel, val_rel = next(gss.split(X_tr_full, y_tr_full, groups=g_tr_full))
        X_tr, y_tr = X_tr_full[tr_rel], y_tr_full[tr_rel]
        X_val, y_val = X_tr_full[val_rel], y_tr_full[val_rel]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)
        X_te_scaled = scaler.transform(X_te)

        model, threshold = train_one_fold(
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_val,
            y_val=y_val,
            model_type=exp["model_type"],
            loss_type=exp["loss_type"],
            use_scheduler=exp["use_scheduler"],
            tune_threshold=exp["tune_threshold"],
        )

        test_loader = DataLoader(VoiceDataset(X_te_scaled, y_te), batch_size=128, shuffle=False)
        y_prob, y_true = predict_probs(model, test_loader)
        y_pred = (y_prob >= threshold).astype(int)

        if len(np.unique(y_true)) < 2:
            auc = 0.5
        else:
            auc = roc_auc_score(y_true, y_prob)
        f1 = f1_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        fold_metrics.append((auc, f1, rec, acc))

        print(
            f"Fold {fold}: AUC={auc:.4f}, F1={f1:.4f}, "
            f"Recall={rec:.4f}, Acc={acc:.4f}, Th={threshold:.2f}"
        )

    m = np.array(fold_metrics)
    result = {
        "name": exp["name"],
        "auc_mean": float(m[:, 0].mean()),
        "auc_std": float(m[:, 0].std()),
        "f1_mean": float(m[:, 1].mean()),
        "recall_mean": float(m[:, 2].mean()),
        "acc_mean": float(m[:, 3].mean()),
    }
    print(
        f"Avg -> AUC={result['auc_mean']:.4f} (+/- {result['auc_std']:.4f}), "
        f"F1={result['f1_mean']:.4f}, Recall={result['recall_mean']:.4f}, "
        f"Acc={result['acc_mean']:.4f}"
    )
    return result


if __name__ == "__main__":
    df = load_data()
    features = get_features(df)
    print(f"{df.shape[0]} samples, {len(features)} features, {df['subject'].nunique()} subjects")
    print(f"Class distribution: {dict(df['status'].value_counts())}")
    print(f"Device: {DEVICE}")

    experiments = [
        {
            "name": "A0: Baseline MLP + Weighted BCE",
            "model_type": "mlp",
            "loss_type": "bce",
            "use_scheduler": False,
            "tune_threshold": False,
        },
        {
            "name": "A1: Residual MLP + Focal Loss",
            "model_type": "residual",
            "loss_type": "focal",
            "use_scheduler": False,
            "tune_threshold": False,
        },
        {
            "name": "A2: Residual MLP + Hybrid Loss + Scheduler + Threshold Tuning",
            "model_type": "residual",
            "loss_type": "hybrid",
            "use_scheduler": True,
            "tune_threshold": True,
        },
    ]

    all_results = []
    for exp in experiments:
        all_results.append(evaluate_experiment(df, exp))

    print("\n=== Ablation Ranking by Mean AUC ===")
    ranked = sorted(all_results, key=lambda x: x["auc_mean"], reverse=True)
    for i, r in enumerate(ranked, start=1):
        print(
            f"{i}. {r['name']} -> AUC={r['auc_mean']:.4f}, F1={r['f1_mean']:.4f}, "
            f"Recall={r['recall_mean']:.4f}, Acc={r['acc_mean']:.4f}"
        )