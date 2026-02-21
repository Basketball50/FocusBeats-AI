import argparse
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
DEMUC_DIR = ROOT / "data" / "demucs_out" / "htdemucs"
OUT_DIR = ROOT / "data" / "transformed_instrumental"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_mono(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    return y.astype(np.float32), sr


def main():
    parser = argparse.ArgumentParser(
        description="Build a clean instrumental (drums + bass + other) from Demucs stems, dropping vocals."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to original WAV (relative to project root or absolute)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional explicit output path (default: data/transformed_instrumental/<name>_instrumental.wav)",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = ROOT / in_path

    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    song_dir = DEMUC_DIR / in_path.stem
    if not song_dir.exists():
        raise SystemExit(
            f"Demucs stems not found for '{in_path.stem}' in {song_dir}.\n"
            "Run Demucs first from project root (inside demucs_env):\n\n"
            f"  python -m demucs -n htdemucs -o data/demucs_out \"{in_path.relative_to(ROOT)}\""
        )

    stem_names = ["drums", "bass", "other"]
    stems = []
    sr_ref = None

    for name in stem_names:
        stem_path = song_dir / f"{name}.wav"
        if not stem_path.exists():
            print(f"[WARN] Missing stem: {stem_path}, skipping this stem.")
            continue

        y, sr = load_mono(stem_path)

        if sr_ref is None:
            sr_ref = sr
        elif sr != sr_ref:
            y = librosa.resample(y, orig_sr=sr, target_sr=sr_ref)

        stems.append(y)

    if not stems:
        raise SystemExit(f"No usable stems found in {song_dir}")

    min_len = min(len(s) for s in stems)
    stems = [s[:min_len] for s in stems]

    mix = np.zeros(min_len, dtype=np.float32)
    for s in stems:
        mix += s

    max_val = float(np.max(np.abs(mix)))
    if max_val > 0:
        mix = 0.97 * mix / max_val

    if args.out is None:
        out_name = in_path.stem + "_instrumental.wav"
        out_path = OUT_DIR / out_name
    else:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = ROOT / out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, mix, sr_ref)
    print(f"[INFO] Wrote instrumental (no vocals) to: {out_path}")


if __name__ == "__main__":
    main()
