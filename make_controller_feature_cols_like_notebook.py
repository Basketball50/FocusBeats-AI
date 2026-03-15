from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def norm_track(s: str) -> str:
    if pd.isna(s):
        return ""
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
    num_cols: list[str] = []
    tmp_df = df.copy()
    for c in tmp_df.columns:
        if c in NON_FEATURE:
            continue
        if pd.api.types.is_numeric_dtype(tmp_df[c]):
            num_cols.append(c)
        else:
            s = pd.to_numeric(tmp_df[c], errors="coerce")
            if s.notna().mean() > 0.95:
                tmp_df[c] = s
                num_cols.append(c)
    return sorted(num_cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--ctrl_csv", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    train_path = Path(args.train_csv).resolve()
    ctrl_path = Path(args.ctrl_csv).resolve()
    out_path = Path(args.out_json).resolve()

    train = pd.read_csv(train_path)
    ctrl = pd.read_csv(ctrl_path)

    train = train.copy()
    ctrl = ctrl.copy()
    train["track_key"] = train["track"].map(norm_track)
    ctrl["track_key"] = ctrl["track"].map(norm_track)

    merged = train.merge(ctrl.drop(columns=["track"]), on="track_key", how="left")

    feature_cols = infer_feature_cols(merged)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(feature_cols, indent=2))

    knob_like = [c for c in feature_cols if any(k in c for k in ["low_gain_db","mid_gain_db","high_gain_db","transient","drc","best_"])]
    print("[DONE] wrote:", out_path)
    print("n_feature_cols:", len(feature_cols))
    print("knob_like_examples:", knob_like[:20])


if __name__ == "__main__":
    main()
