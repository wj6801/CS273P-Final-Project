import warnings
warnings.filterwarnings("ignore")

import numpy as np
from itertools import product
from ucimlrepo import fetch_ucirepo
from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score

SEED = 42
np.random.seed(SEED)


def load_data():
    parkinsons = fetch_ucirepo(id=174)
    df = parkinsons.data.original.copy()
    df["subject"] = df["name"].str.extract(r"(R\d+_S\d+)")[0]
    return df


def get_features(df):
    return [c for c in df.columns if c not in ("name", "status", "subject")]


def evaluate_model(model_name, model, X, y, groups, use_scaler):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    fold_aucs = []
    fold_f1 = []
    fold_recall = []
    fold_acc = []

    print(f"\n=== {model_name} ===")
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if use_scaler:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        fold_aucs.append(auc)
        fold_f1.append(f1)
        fold_recall.append(rec)
        fold_acc.append(acc)

        print(
            f"Fold {fold}: "
            f"AUC={auc:.4f}, F1={f1:.4f}, Recall={rec:.4f}, Acc={acc:.4f}"
        )

    result = {
        "model": model_name,
        "auc_mean": np.mean(fold_aucs),
        "auc_std": np.std(fold_aucs),
        "f1_mean": np.mean(fold_f1),
        "recall_mean": np.mean(fold_recall),
        "acc_mean": np.mean(fold_acc),
    }

    print(
        f"Avg -> AUC={result['auc_mean']:.4f} (+/- {result['auc_std']:.4f}), "
        f"F1={result['f1_mean']:.4f}, Recall={result['recall_mean']:.4f}, "
        f"Acc={result['acc_mean']:.4f}"
    )
    return result


def cv_auc_for_params(make_model, params, X, y, groups, use_scaler):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_aucs = []

    for train_idx, test_idx in sgkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if use_scaler:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model = make_model(params)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        fold_aucs.append(roc_auc_score(y_test, y_prob))

    return float(np.mean(fold_aucs))


def tune_model(model_name, make_model, param_grid, X, y, groups, use_scaler):
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]

    best_params = None
    best_auc = -1.0

    print(f"\n--- Tuning {model_name} ---")
    trial = 1
    for combo in product(*values):
        params = dict(zip(keys, combo))
        mean_auc = cv_auc_for_params(make_model, params, X, y, groups, use_scaler)

        print(f"Trial {trial}: params={params} -> mean AUC={mean_auc:.4f}")
        trial += 1

        if mean_auc > best_auc:
            best_auc = mean_auc
            best_params = params

    print(f"Best for {model_name}: params={best_params}, mean AUC={best_auc:.4f}")
    return best_params, best_auc


def evaluate_soft_voting_ensemble(X, y, groups, lr_params, svm_params, xgb_params):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    fold_aucs = []
    fold_f1 = []
    fold_recall = []
    fold_acc = []

    print("\n=== 4) Soft Voting Ensemble (LR + SVM + XGBoost) ===")
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups), start=1):
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        lr = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=SEED,
            **lr_params,
        )
        svm = SVC(
            probability=True,
            class_weight="balanced",
            random_state=SEED,
            **svm_params,
        )
        xgb = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=SEED,
            **xgb_params,
        )

        lr.fit(X_train_scaled, y_train)
        svm.fit(X_train_scaled, y_train)
        xgb.fit(X_train_raw, y_train)

        lr_prob = lr.predict_proba(X_test_scaled)[:, 1]
        svm_prob = svm.predict_proba(X_test_scaled)[:, 1]
        xgb_prob = xgb.predict_proba(X_test_raw)[:, 1]

        y_prob = (lr_prob + svm_prob + xgb_prob) / 3.0
        y_pred = (y_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        fold_aucs.append(auc)
        fold_f1.append(f1)
        fold_recall.append(rec)
        fold_acc.append(acc)

        print(
            f"Fold {fold}: "
            f"AUC={auc:.4f}, F1={f1:.4f}, Recall={rec:.4f}, Acc={acc:.4f}"
        )

    result = {
        "model": "4) Soft Voting Ensemble",
        "auc_mean": np.mean(fold_aucs),
        "auc_std": np.std(fold_aucs),
        "f1_mean": np.mean(fold_f1),
        "recall_mean": np.mean(fold_recall),
        "acc_mean": np.mean(fold_acc),
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

    X = df[features].values
    y = df["status"].values.astype(int)
    groups = df["subject"].values

    print("Parkinsons dataset (UCI id=174)")
    print(f"Samples: {len(df)}")
    print(f"Features: {len(features)}")
    print(f"Subjects: {df['subject'].nunique()}")
    print(f"Class distribution: {dict(df['status'].value_counts())}")

    neg = (y == 0).sum()
    pos = (y == 1).sum()
    scale_pos_weight = neg / pos

    lr_grid = {
        "C": [0.3, 1.0, 3.0, 10.0],
        "solver": ["liblinear", "lbfgs"],
    }
    svm_grid = {
        "kernel": ["rbf"],
        "C": [0.5, 1.0, 2.0, 5.0],
        "gamma": ["scale", 0.05, 0.1],
    }
    xgb_grid = {
        "n_estimators": [200, 350, 500],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.8, 0.9],
        "colsample_bytree": [0.8, 0.9],
    }

    def make_lr(params):
        return LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=SEED,
            **params,
        )

    def make_svm(params):
        return SVC(
            probability=True,
            class_weight="balanced",
            random_state=SEED,
            **params,
        )

    def make_xgb(params):
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=SEED,
            **params,
        )

    best_lr_params, _ = tune_model(
        "Logistic Regression", make_lr, lr_grid, X, y, groups, use_scaler=True
    )
    best_svm_params, _ = tune_model(
        "SVM (RBF)", make_svm, svm_grid, X, y, groups, use_scaler=True
    )
    best_xgb_params, _ = tune_model(
        "XGBoost", make_xgb, xgb_grid, X, y, groups, use_scaler=False
    )

    all_results = []
    models = [
        ("1) Logistic Regression (tuned)", make_lr(best_lr_params), True),
        ("2) SVM (RBF) (tuned)", make_svm(best_svm_params), True),
        ("3) XGBoost (tuned)", make_xgb(best_xgb_params), False),
    ]
    for model_name, model, use_scaler in models:
        result = evaluate_model(model_name, model, X, y, groups, use_scaler)
        all_results.append(result)

    ensemble_result = evaluate_soft_voting_ensemble(
        X, y, groups, best_lr_params, best_svm_params, best_xgb_params
    )
    all_results.append(ensemble_result)

    print("\n=== Final Ranking by Mean AUC ===")
    ranked = sorted(all_results, key=lambda d: d["auc_mean"], reverse=True)
    for i, r in enumerate(ranked, start=1):
        print(
            f"{i}. {r['model']}: "
            f"AUC={r['auc_mean']:.4f}, F1={r['f1_mean']:.4f}, "
            f"Recall={r['recall_mean']:.4f}, Acc={r['acc_mean']:.4f}"
        )
