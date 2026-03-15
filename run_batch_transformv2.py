from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".m4a", ".ogg"}


def list_audio_files(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS])


def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def flatten_used_layers(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        return ";".join(str(s) for s in x)
    return str(x)


@dataclass
class RunResult:
    ok: bool
    input_path: Path
    output_path: Path
    meta_path: Path
    error: str = ""


def run_transform_one_v2(
    input_path: Path,
    output_path: Path,
    lofi_layers_dir: Path,
    trim_seconds: float,
    controller_model: Optional[Path],
    controller_cols: Optional[Path],
    ctrl_lookup_csv: Optional[Path],
    demucs_required: bool,
    overwrite: bool,
) -> RunResult:
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")

    if output_path.exists() and meta_path.exists() and not overwrite:
        return RunResult(True, input_path, output_path, meta_path, error="")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "scripts.transform_onev2",
        "--in",
        str(input_path),
        "--out",
        str(output_path),
        "--write_meta",
        "--lofi_layers_dir",
        str(lofi_layers_dir),
        "--trim_seconds",
        str(trim_seconds),
    ]

    if controller_model is not None:
        cmd += ["--controller_model", str(controller_model)]
    if controller_cols is not None:
        cmd += ["--controller_cols", str(controller_cols)]
    if ctrl_lookup_csv is not None:
        cmd += ["--ctrl_lookup_csv", str(ctrl_lookup_csv)]
    if demucs_required:
        cmd += ["--demucs_required"]

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "unknown error").strip()
        return RunResult(False, input_path, output_path, meta_path, error=err[:3000])

    if not meta_path.exists():
        return RunResult(
            False,
            input_path,
            output_path,
            meta_path,
            error="Missing sidecar meta.json (expected --write_meta to produce it).",
        )

    return RunResult(True, input_path, output_path, meta_path, error="")


def parse_meta(meta_path: Path) -> Dict[str, Any]:
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return {}


def write_run_metadata(out_dir: Path, args: argparse.Namespace, summary: Dict[str, Any]) -> None:
    payload = {
        "timestamp_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.executable,
        "python_version": sys.version,
        "cwd": str(ROOT),
        "argv": sys.argv,
        "args": {
            "in_dir": args.in_dir,
            "out_dir": args.out_dir,
            "manifest": args.manifest,
            "lofi_layers_dir": args.lofi_layers_dir,
            "trim_seconds": args.trim_seconds,
            "limit": args.limit,
            "controller_model": args.controller_model,
            "controller_cols": args.controller_cols,
            "ctrl_lookup_csv": args.ctrl_lookup_csv,
            "demucs_required": bool(args.demucs_required),
            "overwrite": bool(args.overwrite),
        },
        "summary": summary,
    }

    out_path = out_dir / "RUN_METADATA.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Batch-run transform_onev2.py and write a reproducible manifest.csv.")
    ap.add_argument("--in_dir", required=True, help="Directory containing input audio files (recursively).")
    ap.add_argument("--out_dir", required=True, help="Directory to write transformed audio outputs.")
    ap.add_argument("--manifest", default=None, help="Path to manifest CSV (default: <out_dir>/manifest.csv).")

    ap.add_argument("--lofi_layers_dir", default=str(ROOT / "data" / "lofi_layers_prepped"))
    ap.add_argument("--trim_seconds", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0, help="If >0, only process first N files (for quick test).")

    ap.add_argument("--controller_model", default=None, help="Optional override for controller model path.")
    ap.add_argument("--controller_cols", default=None, help="Optional override for controller cols path.")

    ap.add_argument(
        "--ctrl_lookup_csv",
        default=str(ROOT / "data" / "eval" / "controller_dataset_yamnet_m003.csv"),
        help="CSV used to merge controller_dataset_yamnet_m003 features by track (notebook behavior).",
    )

    ap.add_argument("--demucs_required", action="store_true", help="Fail if Demucs isn't available/works.")
    ap.add_argument("--overwrite", action="store_true", help="Recompute even if output + meta already exist.")

    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    lofi_layers_dir = Path(args.lofi_layers_dir)

    if not in_dir.is_absolute():
        in_dir = ROOT / in_dir
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if not lofi_layers_dir.is_absolute():
        lofi_layers_dir = ROOT / lofi_layers_dir

    if not in_dir.exists():
        raise SystemExit(f"Input dir not found: {in_dir}")
    if not lofi_layers_dir.exists():
        raise SystemExit(f"Lofi layers dir not found: {lofi_layers_dir}")

    controller_model = Path(args.controller_model) if args.controller_model else None
    controller_cols = Path(args.controller_cols) if args.controller_cols else None
    ctrl_lookup_csv = Path(args.ctrl_lookup_csv) if args.ctrl_lookup_csv else None

    if controller_model and not controller_model.is_absolute():
        controller_model = ROOT / controller_model
    if controller_cols and not controller_cols.is_absolute():
        controller_cols = ROOT / controller_cols
    if ctrl_lookup_csv and not ctrl_lookup_csv.is_absolute():
        ctrl_lookup_csv = ROOT / ctrl_lookup_csv

    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest) if args.manifest else (out_dir / "manifest.csv")
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    files = list_audio_files(in_dir)
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    fieldnames = [
        "input_path",
        "output_path",
        "meta_path",
        "ok",
        "error",
        "focus_before",
        "focus_after",
        "focus_delta",
        "yamnet_similarity",
        "lofi_used",
        "lofi_amount_used",
        "lofi_seed",
        "lofi_layers_dir",
        "lofi_used_layers",
        "lofi_used_layers_count",
        "vocals_removed_method",
        "controller_score",
    ]

    rows: List[Dict[str, Any]] = []
    ok_count = 0
    fail_count = 0

    for idx, inp in enumerate(files, start=1):
        rel = inp.relative_to(in_dir)
        out_path = out_dir / rel.with_suffix(".wav")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        res = run_transform_one_v2(
            input_path=inp,
            output_path=out_path,
            lofi_layers_dir=lofi_layers_dir,
            trim_seconds=float(args.trim_seconds),
            controller_model=controller_model,
            controller_cols=controller_cols,
            ctrl_lookup_csv=ctrl_lookup_csv,
            demucs_required=bool(args.demucs_required),
            overwrite=bool(args.overwrite),
        )

        row: Dict[str, Any] = {k: "" for k in fieldnames}
        row["input_path"] = str(inp)
        row["output_path"] = str(out_path)
        row["meta_path"] = str(res.meta_path)
        row["ok"] = "1" if res.ok else "0"
        row["error"] = res.error

        if res.ok:
            ok_count += 1
            payload = parse_meta(res.meta_path)
            meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
            metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
            knobs = payload.get("knobs", {}) if isinstance(payload, dict) else {}

            row["focus_before"] = safe_get(metrics, ["focus_before"], "")
            row["focus_after"] = safe_get(metrics, ["focus_after"], "")
            row["focus_delta"] = safe_get(metrics, ["focus_delta"], "")
            row["yamnet_similarity"] = safe_get(metrics, ["yamnet_similarity"], "")

            row["lofi_used"] = safe_get(meta, ["lofi_used"], "")
            row["lofi_amount_used"] = safe_get(meta, ["lofi_amount_used"], "")
            row["lofi_seed"] = safe_get(meta, ["lofi_seed"], "")
            row["lofi_layers_dir"] = safe_get(meta, ["lofi_layers_dir"], "")

            lofi_used_layers = safe_get(meta, ["lofi_used_layers"], "")
            lofi_used_layers_count = safe_get(meta, ["lofi_used_layers_count"], "")

            if not lofi_used_layers:
                lofi_used_layers = flatten_used_layers(safe_get(meta, ["used_layers"], ""))
            if lofi_used_layers_count == "" and isinstance(safe_get(meta, ["lofi_used_layers_list"], None), list):
                lofi_used_layers_count = len(safe_get(meta, ["lofi_used_layers_list"], []))

            row["lofi_used_layers"] = lofi_used_layers
            row["lofi_used_layers_count"] = lofi_used_layers_count

            row["vocals_removed_method"] = safe_get(meta, ["vocals_removed_method"], "")
            row["controller_score"] = safe_get(knobs, ["controller_score"], "")

        else:
            fail_count += 1

        rows.append(row)

        if idx % 10 == 0 or idx == len(files):
            print(f"[INFO] {idx}/{len(files)} processed. Latest ok={res.ok}")

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary = {
        "n_total": len(files),
        "n_ok": ok_count,
        "n_failed": fail_count,
        "manifest_path": str(manifest_path),
    }
    write_run_metadata(out_dir, args, summary)

    print(f"[INFO] Wrote manifest CSV: {manifest_path}")
    print(f"[INFO] Wrote run metadata: {out_dir / 'RUN_METADATA.json'}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()

