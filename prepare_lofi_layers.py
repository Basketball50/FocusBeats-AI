import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def safe_stem(name: str) -> str:
    return "".join(c if (c.isalnum() or c in " -_()") else "_" for c in name)


def rms_db(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        v = np.sqrt(np.mean(y * y) + 1e-12)
    else:
        v = np.sqrt(np.mean(y * y) + 1e-12)
    return float(20.0 * np.log10(v + 1e-12))


def normalize_to_dbfs(y: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    cur = rms_db(y)
    gain_db = target_dbfs - cur
    gain = 10 ** (gain_db / 20.0)
    return (y * gain).astype(np.float32)


def trim_silence(y: np.ndarray, top_db: float = 35.0) -> np.ndarray:
    if y.ndim == 1:
        yt, _ = librosa.effects.trim(y, top_db=top_db)
        return yt.astype(np.float32)

    mono = np.mean(y, axis=0)
    _, idx = librosa.effects.trim(mono, top_db=top_db)
    start, end = idx
    return y[:, start:end].astype(np.float32)


def ensure_channels_first(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 1:
        return y.astype(np.float32)
    if y.shape[0] in (1, 2) and y.shape[0] < y.shape[1]:
        return np.transpose(y, (1, 0)).astype(np.float32)
    return y.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="data/lofi_layers_raw")
    parser.add_argument("--out_dir", type=str, default="data/lofi_layers_prepped")
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--target_dbfs", type=float, default=-20.0)
    parser.add_argument("--trim_db", type=float, default=35.0)
    parser.add_argument("--max_seconds", type=float, default=30.0)  
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    in_dir = (root / args.in_dir).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".m4a", ".ogg"}
    files = sorted([p for p in in_dir.rglob("*") if p.suffix.lower() in exts])

    print(f"[INFO] Found {len(files)} lofi layers in {in_dir}")
    print(f"[INFO] Writing prepped layers to {out_dir} (sr={args.sr}, mono=False)")

    if not files:
        return

    for p in files:
        try:
            y, sr = librosa.load(str(p), sr=args.sr, mono=False)

            if args.max_seconds and args.max_seconds > 0:
                max_n = int(args.max_seconds * args.sr)
                if y.ndim == 1:
                    y = y[:max_n]
                else:
                    y = y[:, :max_n]

            y = trim_silence(y, top_db=args.trim_db)
            y = normalize_to_dbfs(y, target_dbfs=args.target_dbfs)

            y_out = ensure_channels_first(y)

            out_name = f"{safe_stem(p.stem)}_prep.wav"
            out_path = out_dir / out_name

            sf.write(
                str(out_path),
                y_out.astype(np.float32),
                args.sr,
                format="WAV",
                subtype="PCM_16",
            )

            print(f"  [OK] {p.name} -> {out_path.name}")

        except Exception as e:
            print(f"  [WARN] Failed on {p.name}: {e}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
