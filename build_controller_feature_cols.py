from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd


DEFAULT_DROP_PREFIXES = (
    "best_",          
    "focus_",         
    "sim_",          
    "score_",        
)

DEFAULT_DROP_COLS = {
    "track",
    "base_best_path",
    "used_layers",
    "chosen_type",
    "use_lofi",
    "delta",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/eval/controller_dataset_yamnet.csv")
    ap.add_argument("--out_json", default="models/controller_feature_cols.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    csv_path = root / args.csv
    out_path = root / args.out_json

    df = pd.read_csv(csv_path)

    cols = list(df.columns)

    keep = []
    for c in cols:
        if c in DEFAULT_DROP_COLS:
            continue
        if any(str(c).startswith(p) for p in DEFAULT_DROP_PREFIXES):
            continue
        if c.endswith("_path"):
            continue
        if c.endswith("_id"):
            continue
        keep.append(c)

    numericish = []
    for c in keep:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() >= 0.50:  
            numericish.append(c)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(numericish, indent=2))

    print("[DONE] wrote:", out_path)
    print("cols_total:", len(cols))
    print("cols_kept_numericish:", len(numericish))


if __name__ == "__main__":
    main()
