import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
from ucimlrepo import fetch_ucirepo
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

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


if __name__ == "__main__":
    df = load_data()
    features = get_features(df)
    print(f"  {df.shape[0]} samples, {len(features)} features, {df['subject'].nunique()} subjects")
    print(f"  Class distribution: {dict(df['status'].value_counts())}")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(sgkf.split(df[features].values, df["status"].values, df["subject"].values))

    input_dim = len(features)
    print(f"\n  Input dim: {input_dim}")