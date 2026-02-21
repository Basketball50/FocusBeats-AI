import argparse
import csv
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

W_FOCUS = 0.7
W_SIM = 0.3


def f(x):
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return None
        return float(s)
    except Exception:
        return None


def score(focus, sim):
    return W_FOCUS * focus + W_SIM * sim


def sanitize_filename(name: str) -> str:
    bad = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    out = name
    for b in bad:
        out = out.replace(b, "_")
    return out


def audio_features(path, sr=22050):
    y, sr = librosa.load(path, sr=sr, mono=True)
    dur_s = max(1e-6, float(len(y)) / float(sr))

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])

    rms = float(np.mean(librosa.feature.rms(y=y)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_rate = float(len(onsets) / dur_s)

    return {
        "tempo": tempo,
        "rms": rms,
        "centroid": centroid,
        "flatness": flatness,
        "onset_rate": onset_rate,
    }


def load_yamnet():
    print("[INFO] Loading YAMNet...")
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
    print("[INFO] YAMNet loaded.")
    return yamnet


def yamnet_mean_embedding_from_wav(path: Path, yamnet):
    y16, _ = librosa.load(path, sr=16000, mono=True)
    waveform = tf.convert_to_tensor(y16.astype(np.float32), dtype=tf.float32)
    _, embeddings, _ = yamnet(waveform)
    emb = embeddings.numpy()
    return emb.mean(axis=0).astype(np.float32)  


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best_versions", default="data/eval/best_versions_alltracks.csv")
    ap.add_argument("--best_lofi_eval", default="data/eval/best_lofi_eval.csv")
    ap.add_argument("--out", default="data/eval/controller_dataset.csv")
    ap.add_argument(
        "--base_best_dir",
        default="data/transformed_best_alltracks",
        help="Directory containing {track}_best.wav files (the base audio we are gating on).",
    )
    ap.add_argument("--use_yamnet_emb", action="store_true")
    ap.add_argument(
        "--yamnet_on",
        choices=["base_best", "original", "both"],
        default="base_best",
        help="Which audio to embed for yamnet_* features. Default: base_best (recommended).",
    )

    ap.add_argument(
        "--margin",
        type=float,
        default=0.01,
        help="Tie margin. If --drop_ties, drop rows with |delta| <= margin.",
    )
    ap.add_argument(
        "--drop_ties",
        action="store_true",
        help="Drop rows with |score_lofi-score_base| <= margin (removes near-ties / label noise).",
    )

    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    best_versions = root / args.best_versions
    best_lofi_eval = root / args.best_lofi_eval
    out_path = root / args.out
    base_best_dir = root / args.base_best_dir
    general_dir = root / "data" / "processed" / "general"

    if not best_versions.exists():
        raise SystemExit(f"Missing: {best_versions}")
    if not best_lofi_eval.exists():
        raise SystemExit(f"Missing: {best_lofi_eval}")
    if not base_best_dir.exists():
        raise SystemExit(f"Missing base_best_dir: {base_best_dir}")

    lofi_map = {}
    with open(best_lofi_eval, "r", newline="") as f_in:
        r = csv.DictReader(f_in)
        required = {"track", "focus_base", "sim_base", "focus_lofi", "sim_lofi"}
        missing_cols = required - set(r.fieldnames or [])
        if missing_cols:
            raise SystemExit(f"best_lofi_eval missing required columns: {sorted(missing_cols)}")

        for row in r:
            track = (row.get("track") or "").strip()
            if not track:
                continue
            lofi_map[track] = {
                "focus_base": f(row.get("focus_base")),
                "sim_base": f(row.get("sim_base")),
                "focus_lofi": f(row.get("focus_lofi")),
                "sim_lofi": f(row.get("sim_lofi")),
                "used_layers": row.get("used_layers", ""),
            }

    yamnet = load_yamnet() if args.use_yamnet_emb else None

    missing_original_audio = 0
    missing_base_best = 0
    missing_lofi = 0
    bad_metrics = 0
    dropped_ties = 0
    kept_pos = 0
    kept_neg = 0

    rows_out = []
    with open(best_versions, "r", newline="") as f_in:
        r = csv.DictReader(f_in)
        if not (r.fieldnames and "track" in r.fieldnames):
            raise SystemExit("best_versions CSV missing 'track' column")

        for row in r:
            track = (row.get("track") or "").strip()
            if not track:
                continue

            orig_path = general_dir / f"{track}.wav"
            if args.use_yamnet_emb and args.yamnet_on in ("original", "both"):
                if not orig_path.exists():
                    missing_original_audio += 1
                    continue

            base_best_name = sanitize_filename(f"{track}_best.wav")
            base_best_path = base_best_dir / base_best_name
            if not base_best_path.exists():
                missing_base_best += 1
                continue

            if track not in lofi_map:
                missing_lofi += 1
                continue

            feats = audio_features(base_best_path)

            fb = lofi_map[track]["focus_base"]
            sb = lofi_map[track]["sim_base"]
            fl = lofi_map[track]["focus_lofi"]
            sl = lofi_map[track]["sim_lofi"]
            if fb is None or sb is None or fl is None or sl is None:
                bad_metrics += 1
                continue

            score_base = score(fb, sb)
            score_lofi = score(fl, sl)

            delta = score_lofi - score_base

            if args.drop_ties and abs(delta) <= args.margin:
                dropped_ties += 1
                continue

            use_lofi = 1 if delta > 0 else 0
            if use_lofi == 1:
                kept_pos += 1
            else:
                kept_neg += 1

            out_row = {
                "track": track,
                "base_best_path": str(base_best_path.relative_to(root)),
                **feats,
                "focus_base": fb,
                "sim_base": sb,
                "score_base": score_base,
                "focus_lofi": fl,
                "sim_lofi": sl,
                "score_lofi": score_lofi,
                "delta": float(delta),
                "use_lofi": use_lofi,
                "used_layers": lofi_map[track].get("used_layers", ""),
                "chosen_type": row.get("chosen_type", ""),
            }

            for k, v in row.items():
                if k.startswith("best_"):
                    out_row[k] = v

            if args.use_yamnet_emb:
                if args.yamnet_on in ("base_best", "both"):
                    emb_base = yamnet_mean_embedding_from_wav(base_best_path, yamnet)
                    for i, val in enumerate(emb_base.tolist()):
                        out_row[f"yamnet_{i}"] = val

                if args.yamnet_on in ("original", "both"):
                    emb_orig = yamnet_mean_embedding_from_wav(orig_path, yamnet)
                    for i, val in enumerate(emb_orig.tolist()):
                        out_row[f"yamnet_orig_{i}"] = val

            rows_out.append(out_row)

    if not rows_out:
        raise SystemExit(
            "[ERR] No rows produced.\n"
            f"Missing original audio: {missing_original_audio}, Missing base_best wav: {missing_base_best}, "
            f"Missing lofi rows: {missing_lofi}, Bad metrics: {bad_metrics}, Dropped ties: {dropped_ties}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows_out[0].keys())
    with open(out_path, "w", newline="") as f_out:
        w = csv.DictWriter(f_out, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    print("[INFO] Wrote:", out_path)
    print("[INFO] Rows:", len(rows_out))
    print("[INFO] Dropped ties:", dropped_ties, f"(margin={args.margin}, drop_ties={bool(args.drop_ties)})")
    print("[INFO] Kept pos/neg:", kept_pos, kept_neg, f"(pos_rate={kept_pos/max(1,(kept_pos+kept_neg)):.3f})")
    print("[INFO] Missing original audio:", missing_original_audio)
    print("[INFO] Missing base_best wav:", missing_base_best)
    print("[INFO] Missing lofi rows:", missing_lofi)
    print("[INFO] Bad metrics rows skipped:", bad_metrics)


if __name__ == "__main__":
    main()
