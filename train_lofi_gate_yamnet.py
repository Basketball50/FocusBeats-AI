import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
)
import joblib


BASE_AUDIO_FEATURES = ["tempo", "rms", "centroid", "flatness", "onset_rate"]
EXTRA_FEATURES = ["focus_base", "sim_base", "score_base", "chosen_type_is_dsp_grid"]


def pick_threshold(y_true, p):
    best_t = 0.5
    best_v = -1.0
    for t in np.linspace(0.05, 0.95, 19):
        yhat = (p >= t).astype(int)
        v = balanced_accuracy_score(y_true, yhat)
        if v > best_v:
            best_v = v
            best_t = float(t)
    return best_t, best_v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/eval/controller_dataset_yamnet.csv")
    ap.add_argument("--outdir", default="models/lofi_gate_yamnet")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pca", type=int, default=32, help="PCA components for yamnet embeddings. 0 disables PCA.")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    csv_path = root / args.csv
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    if "chosen_type" in df.columns:
        df["chosen_type_is_dsp_grid"] = (df["chosen_type"] == "dsp_grid").astype(int)
    else:
        df["chosen_type_is_dsp_grid"] = 0

    yamnet_cols = [c for c in df.columns if c.startswith("yamnet_") and not c.startswith("yamnet_orig_")]
    if len(yamnet_cols) != 1024:
        raise SystemExit(f"Expected 1024 yamnet_* cols (base_best), found {len(yamnet_cols)}")

    features = BASE_AUDIO_FEATURES + EXTRA_FEATURES + yamnet_cols

    df = df.dropna(subset=features + ["use_lofi"]).copy()
    if len(df) < 10:
        raise SystemExit(f"Too few rows after dropna: {len(df)} (need more data).")

    y_all = df["use_lofi"].to_numpy(dtype=np.int64)
    if len(np.unique(y_all)) < 2:
        raise SystemExit("Only one class present in use_lofi after filtering; cannot train classifier.")

    train_idx, test_idx = train_test_split(
        df.index.to_numpy(),
        test_size=0.2,
        random_state=args.seed,
        stratify=y_all,
    )
    df_tr = df.loc[train_idx]
    df_te = df.loc[test_idx]
    y_tr = df_tr["use_lofi"].to_numpy(dtype=np.int64)
    y_te = df_te["use_lofi"].to_numpy(dtype=np.int64)

    pca_k = int(args.pca) if args.pca else 0
    if pca_k > 0:
        n_train = len(df_tr)
        max_k = min(n_train - 1, len(yamnet_cols))  
        if max_k < 1:
            print("[WARN] Train set too small for PCA; disabling PCA.")
            pca_k = 0
        elif pca_k > max_k:
            print(f"[WARN] PCA={pca_k} too large for train n={n_train}. Capping PCA to {max_k}.")
            pca_k = max_k

    if pca_k > 0:
        yamnet_transform = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_k, random_state=args.seed)),
        ])
    else:
        yamnet_transform = Pipeline([
            ("scaler", StandardScaler()),
        ])

    pre = ColumnTransformer(
        transformers=[
            ("base", StandardScaler(), BASE_AUDIO_FEATURES + EXTRA_FEATURES),
            ("yamnet", yamnet_transform, yamnet_cols),
        ],
        remainder="drop",
    )

    model = Pipeline(steps=[
        ("pre", pre),
        ("clf", LogisticRegression(
            max_iter=8000,
            class_weight="balanced",
        )),
    ])

    model.fit(df_tr[features], y_tr)
    p_te = model.predict_proba(df_te[features])[:, 1]

    t_best, bacc_best = pick_threshold(y_te, p_te)
    yhat = (p_te >= t_best).astype(int)

    acc = accuracy_score(y_te, yhat)
    bacc = balanced_accuracy_score(y_te, yhat)
    auc = roc_auc_score(y_te, p_te) if len(np.unique(y_te)) > 1 else float("nan")

    print("[INFO] rows:", len(df))
    print("[INFO] pos rate:", round(float(df["use_lofi"].mean()), 4))
    print("[INFO] PCA used:", pca_k)
    print("[INFO] chosen threshold:", round(t_best, 3), f"(best bAcc={bacc_best:.4f})")
    print("[OK] Test accuracy:", round(acc, 4))
    print("[OK] Test balanced accuracy:", round(bacc, 4))
    print("[OK] Test ROC-AUC:", round(auc, 4))
    print("[INFO] Confusion matrix [[tn, fp], [fn, tp]]:")
    print(confusion_matrix(y_te, yhat))
    print(classification_report(y_te, yhat, digits=4))

    joblib.dump(
        {
            "model": model,
            "features": features,
            "yamnet_cols": yamnet_cols,
            "base_cols": BASE_AUDIO_FEATURES + EXTRA_FEATURES,
            "threshold": t_best,
            "pca_k": pca_k,
        },
        outdir / "model.joblib",
    )
    print("[INFO] Saved:", outdir / "model.joblib")


if __name__ == "__main__":
    main()
