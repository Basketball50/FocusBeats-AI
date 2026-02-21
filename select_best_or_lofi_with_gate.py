from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from scripts.utils.track_paths import (
    sanitize_filename,
    base_best_wav_path,
    lofi_wav_path,
)


def _find_track_column(df: pd.DataFrame) -> str:
    candidates = ["track", "track_name", "name", "title", "id", "track_id", "filename"]
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(
        "Could not find a track column. Expected one of: "
        + ", ".join(candidates)
        + f". Columns present: {list(df.columns)[:30]} ..."
    )


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _dir_has_any(dirpath: Path, pattern: str) -> bool:
    try:
        next(dirpath.glob(pattern))
        return True
    except StopIteration:
        return False


def _auto_detect_audio_dirs(root: Path) -> tuple[Path, Path]:
    data_dir = root / "data"
    if not data_dir.exists():
        raise SystemExit(f"Expected data directory not found: {data_dir}")

    # common names in YOUR repo
    base_candidates = [
        data_dir / "transformed_best_alltracks",
        data_dir / "transformed_best_all_tracks",
        data_dir / "transformed_best",
        data_dir / "transformed_selected",
    ]
    lofi_candidates = [
        data_dir / "transformed_best_lofi",
        data_dir / "transformed_best_lofi_a",
        data_dir / "transformed_best_lofi_b",
        data_dir / "transformed_selected",
    ]

    base_dir = None
    for d in base_candidates:
        if d.exists() and d.is_dir() and _dir_has_any(d, "*_best.wav"):
            base_dir = d
            break

    lofi_dir = None
    for d in lofi_candidates:
        if d.exists() and d.is_dir() and _dir_has_any(d, "*_lofi.wav"):
            lofi_dir = d
            break

    if base_dir is None or lofi_dir is None:
        dirs = [p for p in data_dir.rglob("*") if p.is_dir()]

        if base_dir is None:
            for d in dirs:
                if _dir_has_any(d, "*_best.wav"):
                    base_dir = d
                    break

        if lofi_dir is None:
            for d in dirs:
                if _dir_has_any(d, "*_lofi.wav"):
                    lofi_dir = d
                    break

    if base_dir is None:
        raise SystemExit(
            "Could not auto-detect base_dir (folder with '*_best.wav') under data/. "
            "Pass it explicitly with --base_dir."
        )
    if lofi_dir is None:
        raise SystemExit(
            "Could not auto-detect lofi_dir (folder with '*_lofi.wav') under data/. "
            "Pass it explicitly with --lofi_dir."
        )

    return base_dir, lofi_dir


def _synthesize_missing_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    df = df.copy()

    if "chosen_type_is_dsp_grid" in features and "chosen_type_is_dsp_grid" not in df.columns:
        if "chosen_type" in df.columns:
            df["chosen_type_is_dsp_grid"] = (df["chosen_type"] == "dsp_grid").astype(int)
        else:
            df["chosen_type_is_dsp_grid"] = 0

    return df


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", default="data/eval/controller_dataset_yamnet.csv")
    ap.add_argument("--model", default="models/lofi_gate_yamnet/model.joblib")

    ap.add_argument("--base_dir", default="")
    ap.add_argument("--lofi_dir", default="")

    ap.add_argument("--outdir", default="data/transformed_selected")
    ap.add_argument("--limit", type=int, default=0)

    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]

    csv_path = root / args.csv
    model_path = root / args.model
    outdir = root / args.outdir

    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    base_dir = (root / args.base_dir).resolve() if args.base_dir else None
    lofi_dir = (root / args.lofi_dir).resolve() if args.lofi_dir else None

    if base_dir is not None and not base_dir.exists():
        print(f"[WARN] base_dir not found: {base_dir} -> will auto-detect under data/")
        base_dir = None
    if lofi_dir is not None and not lofi_dir.exists():
        print(f"[WARN] lofi_dir not found: {lofi_dir} -> will auto-detect under data/")
        lofi_dir = None

    if base_dir is None or lofi_dir is None:
        auto_base, auto_lofi = _auto_detect_audio_dirs(root)
        if base_dir is None:
            base_dir = auto_base
        if lofi_dir is None:
            lofi_dir = auto_lofi

    if not base_dir.exists():
        raise SystemExit(f"Base dir not found: {base_dir}")
    if not lofi_dir.exists():
        raise SystemExit(f"Lofi dir not found: {lofi_dir}")

    _ensure_dir(outdir)

    pack = joblib.load(model_path)
    model = pack["model"]
    features = pack["features"]
    threshold = float(pack.get("threshold", 0.5))

    df = pd.read_csv(csv_path)
    track_col = _find_track_column(df)

    df = _synthesize_missing_features(df, features)

    missing_cols = [c for c in features if c not in df.columns]
    if missing_cols:
        raise SystemExit(
            "CSV is missing required feature columns expected by model.joblib:\n"
            + "\n".join(missing_cols[:40])
            + ("\n..." if len(missing_cols) > 40 else "")
        )

    df = df.dropna(subset=features + [track_col]).copy()

    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    if len(df) == 0:
        raise SystemExit("No rows to process after dropna/limit.")

    X = df[features]
    p_lofi = model.predict_proba(X)[:, 1]
    choose_lofi = (p_lofi >= threshold).astype(int)

    n_missing_base = 0
    n_missing_lofi = 0
    n_chosen_lofi = 0
    n_chosen_base = 0

    rows_out = []
    positions = {idx: pos for pos, idx in enumerate(df.index.to_numpy())}

    for idx, row in df.iterrows():
        pos = positions[idx]

        track_name = str(row[track_col])
        safe = sanitize_filename(track_name)

        base_path = base_best_wav_path(base_dir, track_name)
        lofi_path = lofi_wav_path(lofi_dir, track_name)

        use_lofi = bool(choose_lofi[pos])
        prob = float(p_lofi[pos])

        if use_lofi:
            src = lofi_path
            if not src.exists():
                n_missing_lofi += 1
                if base_path.exists():
                    src = base_path
                    use_lofi = False
                else:
                    n_missing_base += 1
                    rows_out.append(
                        {
                            "track": track_name,
                            "safe": safe,
                            "p_lofi": prob,
                            "threshold": threshold,
                            "chosen": "MISSING_BOTH",
                            "src": "",
                            "dst": "",
                        }
                    )
                    continue
        else:
            src = base_path
            if not src.exists():
                n_missing_base += 1
                if lofi_path.exists():
                    src = lofi_path
                    use_lofi = True
                else:
                    n_missing_lofi += 1
                    rows_out.append(
                        {
                            "track": track_name,
                            "safe": safe,
                            "p_lofi": prob,
                            "threshold": threshold,
                            "chosen": "MISSING_BOTH",
                            "src": "",
                            "dst": "",
                        }
                    )
                    continue

        chosen_tag = "lofi" if use_lofi else "base"
        dst = outdir / f"{safe}_selected_{chosen_tag}.wav"
        shutil.copy2(src, dst)

        if use_lofi:
            n_chosen_lofi += 1
        else:
            n_chosen_base += 1

        rows_out.append(
            {
                "track": track_name,
                "safe": safe,
                "p_lofi": prob,
                "threshold": threshold,
                "chosen": chosen_tag,
                "src": str(src),
                "dst": str(dst),
            }
        )

    report_path = outdir / "selection_report.csv"
    pd.DataFrame(rows_out).to_csv(report_path, index=False)

    print("[OK] Selection complete.")
    print("[INFO] base_dir:", base_dir)
    print("[INFO] lofi_dir:", lofi_dir)
    print("[INFO] Rows processed:", len(df))
    print("[INFO] Threshold:", threshold)
    print("[INFO] Chosen base:", n_chosen_base)
    print("[INFO] Chosen lofi:", n_chosen_lofi)
    print("[WARN] Missing base encountered:", n_missing_base)
    print("[WARN] Missing lofi encountered:", n_missing_lofi)
    print("[INFO] Report saved:", report_path)
    print("[INFO] Output dir:", outdir)


if __name__ == "__main__":
    main()
