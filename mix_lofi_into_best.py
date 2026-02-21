from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

from scripts.focus_yamnet import load_focus_bundle, load_wav_16k_mono, cosine_similarity
from scripts.lofi_layers import apply_lofi_layers, ensure_stereo_ch_first


def sanitize_filename(name: str) -> str:
    bad = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    out = name
    for b in bad:
        out = out.replace(b, "_")
    return out


def preprocess_for_metrics_16k_mono(y_16k_mono: np.ndarray, target_rms_db: float = -18.0, peak: float = 0.98) -> np.ndarray:
    y = np.asarray(y_16k_mono, dtype=np.float32)
    r = float(np.sqrt(np.mean(y * y) + 1e-12))
    cur_db = 20.0 * np.log10(r + 1e-12)
    gain_db = target_rms_db - cur_db
    gain = 10.0 ** (gain_db / 20.0)
    y = y * gain
    p = float(np.max(np.abs(y))) if y.size else 0.0
    if p > peak:
        y = y * (peak / (p + 1e-12))
    return y.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best_csv", default="data/eval/best_versions_alltracks.csv")
    ap.add_argument("--base_best_dir", default="data/transformed_best_alltracks")
    ap.add_argument("--out_dir", default="data/transformed_best_lofi")
    ap.add_argument("--layers_dir", default="data/lofi_layers_prepped")
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amount", type=float, default=0.55, help="Default lofi amount used for offline render.")
    ap.add_argument("--target_rms_db", type=float, default=-18.0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    best_csv = root / args.best_csv
    base_best_dir = root / args.base_best_dir
    out_dir = root / args.out_dir
    layers_dir = root / args.layers_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not best_csv.exists():
        raise SystemExit(f"Missing: {best_csv}")
    if not base_best_dir.exists():
        raise SystemExit(f"Missing base_best_dir: {base_best_dir}")
    if not layers_dir.exists():
        raise SystemExit(f"Missing layers_dir: {layers_dir}")

    focus = load_focus_bundle(root)
    general_dir = root / "data" / "processed" / "general"

    header = [
        "track",
        "chosen_path",
        "lofi_out_path",
        "focusability_orig",
        "focusability_base",
        "focusability_lofi",
        "sim_base",
        "sim_lofi",
        "used_layers",
        "focus_pos_label",
    ]
    rows_out = []

    with best_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        if not (r.fieldnames and "track" in r.fieldnames):
            raise SystemExit("best_csv missing 'track' column")

        for row in r:
            track = (row.get("track") or "").strip()
            if not track:
                continue

            orig_path = general_dir / f"{track}.wav"
            if not orig_path.exists():
                continue

            chosen_path = base_best_dir / sanitize_filename(f"{track}_best.wav")
            if not chosen_path.exists():
                continue

            y_orig_16k = preprocess_for_metrics_16k_mono(load_wav_16k_mono(orig_path), args.target_rms_db)
            y_base_16k = preprocess_for_metrics_16k_mono(load_wav_16k_mono(chosen_path), args.target_rms_db)

            emb_orig = focus.yamnet_mean_embedding(y_orig_16k)
            emb_base = focus.yamnet_mean_embedding(y_base_16k)

            focus_orig = focus.focusability(emb_orig)
            focus_base = focus.focusability(emb_base)
            sim_base = cosine_similarity(emb_orig, emb_base)

            y_base, _ = librosa.load(str(chosen_path), sr=args.sr, mono=False)
            base2n = ensure_stereo_ch_first(y_base)

            y_lofi_2n, used = apply_lofi_layers(
                base2n=base2n,
                sr=args.sr,
                layers_dir=layers_dir,
                amount=args.amount,
                seed=args.seed + (hash(track) % 10_000),
            )

            out_path = out_dir / sanitize_filename(f"{track}_best_lofi.wav")
            sf.write(str(out_path), np.transpose(y_lofi_2n, (1, 0)), args.sr, format="WAV", subtype="PCM_16")

            y_lofi_16k = preprocess_for_metrics_16k_mono(load_wav_16k_mono(out_path), args.target_rms_db)
            emb_lofi = focus.yamnet_mean_embedding(y_lofi_16k)
            focus_lofi = focus.focusability(emb_lofi)
            sim_lofi = cosine_similarity(emb_orig, emb_lofi)

            rows_out.append([
                track,
                str(chosen_path.relative_to(root)),
                str(out_path.relative_to(root)),
                float(focus_orig),
                float(focus_base),
                float(focus_lofi),
                float(sim_base),
                float(sim_lofi),
                ";".join(used),
                focus.pos_label,
            ])

    out_csv = root / "data" / "eval" / "best_lofi_eval.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows_out)

    print("[INFO] Wrote:", out_csv)
    print("[INFO] Lofi renders in:", out_dir)


if __name__ == "__main__":
    main()
