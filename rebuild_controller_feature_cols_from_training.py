from __future__ import annotations

import argparse, json, re
from pathlib import Path

import numpy as np
import pandas as pd


def norm_track(s: str) -> str:
    if pd.isna(s): return ""
    s = str(s).strip()
    s = s.split("/")[-1].split("\\")[-1]
    s = re.sub(r"\.(wav|mp3|flac|m4a|ogg)$", "", s, flags=re.IGNORECASE)
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s.lower().strip()


NON_FEATURE = {
    "track", "variant_id",
    "label_is_best", "label_apply_dsp_track",
    "focus", "sim", "score",
    "best_score", "second_score", "gap_best_second",
    "is_ambiguous", "is_low_sim_track",
    "rel",
    "track_key",
}


def infer_feature_cols(df: pd.DataFrame) -> list[str]:
    num_cols = []
    df = df.copy()
    for c in df.columns:
        if c in NON_FEATURE:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            tmp = pd.to_numeric(df[c], errors="coerce")
            if tmp.notna().mean() > 0.95:
                df[c] = tmp
                num_cols.append(c)
    return sorted(num_cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", default="data/eval")
    ap.add_argument("--ctrl_csv", default="data/eval/controller_dataset_yamnet_m003.csv")
    ap.add_argument("--train_csv", default="data/eval/dsp_knob_variant_dataset_alltracks_train.csv")
    ap.add_argument("--out_json", default="models/controller_feature_cols_FROM_TRAINING.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    eval_dir = (root / args.eval_dir).resolve()
    ctrl_path = (root / args.ctrl_csv).resolve()
    train_path = (root / args.train_csv).resolve()
    out_path = (root / args.out_json).resolve()

    ctrl = pd.read_csv(ctrl_path)
    train = pd.read_csv(train_path)

    ctrl = ctrl.copy()
    ctrl["track_key"] = ctrl["track"].map(norm_track)

    train = train.copy()
    train["track_key"] = train["track"].map(norm_track)

    merged = train.merge(ctrl.drop(columns=["track"]), on="track_key", how="left")

    feature_cols = infer_feature_cols(merged)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(feature_cols, indent=2))
    print("[DONE] wrote:", out_path)
    print("n_features:", len(feature_cols))
    knob_like = [c for c in feature_cols if any(k in c for k in ["low_gain_db","mid_gain_db","high_gain_db","transient_smooth","drc_strength","vocal_cut","lofi_amount"])]
    print("knob_like_found:", knob_like)


if __name__ == "__main__":
    main()
