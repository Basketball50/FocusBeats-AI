import argparse
import csv
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def f(x, default=None):
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def sanitize_filename(name: str) -> str:
    bad = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    out = name
    for b in bad:
        out = out.replace(b, "_")
    return out


def import_apply_dsp(scripts_dir: Path):
    if str(scripts_dir) not in sys.path:
        sys.path.append(str(scripts_dir))
    try:
        from apply_dsp_knobs import apply_dsp  
    except Exception as e:
        raise SystemExit(
            f"Could not import apply_dsp from {scripts_dir/'apply_dsp_knobs.py'}:\n{e}"
        )
    return apply_dsp


def rms_db(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return -120.0
    r = float(np.sqrt(np.mean(y * y) + 1e-12))
    return 20.0 * np.log10(r + 1e-12)


def peak_db(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return -120.0
    p = float(np.max(np.abs(y))) + 1e-12
    return 20.0 * np.log10(p)


def normalize_rms(y: np.ndarray, target_rms_db: float, silence_floor_db: float = -60.0):
   
    y = np.asarray(y, dtype=np.float32)
    cur = rms_db(y)

    if cur < silence_floor_db:
        return y.astype(np.float32), 0.0

    gain_db = target_rms_db - cur
    gain = 10.0 ** (gain_db / 20.0)
    return (y * gain).astype(np.float32), float(gain_db)


def peak_limit(y: np.ndarray, peak: float = 0.98):
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y.astype(np.float32), False
    p = float(np.max(np.abs(y)))
    if p > peak:
        y = y * (peak / (p + 1e-12))
        return y.astype(np.float32), True
    return y.astype(np.float32), False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--best",
        type=str,
        default="",
        help="Path to best_versions CSV (default: data/eval/best_versions_alltracks.csv)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="",
        help="Output dir for final WAVs (default: data/transformed_best_alltracks)",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Print what would happen without writing audio.",
    )

    parser.add_argument(
        "--target_rms_db",
        type=float,
        default=-18.0,
        help="Normalize rendered audio to this RMS level (dBFS). Default -18.0",
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Disable normalization (NOT recommended).",
    )
    parser.add_argument(
        "--silence_floor_db",
        type=float,
        default=-60.0,
        help="If RMS is below this, skip normalization to avoid boosting silence. Default -60 dBFS.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    scripts_dir = root / "scripts"
    eval_dir = root / "data" / "eval"
    instr_dir = root / "data" / "transformed_instrumental"

    best_csv = Path(args.best) if args.best else (eval_dir / "best_versions_alltracks.csv")
    out_dir = Path(args.outdir) if args.outdir else (root / "data" / "transformed_best_alltracks")

    if not best_csv.exists():
        raise SystemExit(f"Missing best_versions CSV: {best_csv}")

    apply_dsp = import_apply_dsp(scripts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_csv = out_dir / "render_log.csv"
    log_rows = []
    log_header = [
        "track",
        "chosen_type",
        "src_instrumental",
        "dst_out",
        "status",
        "note",
        "rms_in_db",
        "rms_out_db",
        "peak_out_db",
        "gain_db_applied",
        "peak_limited",
    ]

    def write_processed(out_path: Path, y_mono: np.ndarray, sr: int, note: str):
        y = np.asarray(y_mono, dtype=np.float32)
        rms_in = rms_db(y)

        gain_db = 0.0
        if args.no_normalize:
            y2 = y
            note2 = note + " | peak_limit_only"
        else:
            y2, gain_db = normalize_rms(y, args.target_rms_db, silence_floor_db=args.silence_floor_db)
            note2 = note + f" | norm_rms={args.target_rms_db:.1f}dB"

        y3, did_limit = peak_limit(y2, 0.98)
        if did_limit:
            note2 += " + peak_limit"

        rms_out = rms_db(y3)
        peak_out = peak_db(y3)

        sf.write(out_path, y3.astype(np.float32), sr)
        return note2, rms_in, rms_out, peak_out, gain_db, did_limit

    with open(best_csv, "r", newline="") as f_in:
        reader = csv.DictReader(f_in)
        required = {"track", "chosen_type"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"best_versions CSV missing required columns: {sorted(missing)}")

        for row in reader:
            track = row["track"]
            chosen_type = (row.get("chosen_type") or "").strip()

            instr_path = instr_dir / f"{track}_instrumental.wav"
            if not instr_path.exists():
                log_rows.append([track, chosen_type, str(instr_path), "", "FAIL", "missing instrumental wav", "", "", "", "", ""])
                print(f"[FAIL] {track}: missing instrumental at {instr_path}")
                continue

            out_name = sanitize_filename(f"{track}_best.wav")
            out_path = out_dir / out_name

            if args.dryrun:
                print(f"[DRY] {track} -> {out_path.name} (chosen_type={chosen_type})")
                log_rows.append([track, chosen_type, str(instr_path), str(out_path), "DRYRUN", "no audio written", "", "", "", "", ""])
                continue

            try:
                y, sr = librosa.load(instr_path, sr=None, mono=True)

                if chosen_type == "instrumental":
                    print(f"[BASE] {track} -> {out_path.name} (instrumental)")
                    note, rms_in, rms_out, peak_out, gain_db, did_limit = write_processed(out_path, y, sr, "rendered instrumental")

                elif chosen_type == "dsp_grid":
                    low_gain_db = f(row.get("best_low_gain_db"), 0.0)
                    mid_gain_db = f(row.get("best_mid_gain_db"), 0.0)
                    high_gain_db = f(row.get("best_high_gain_db"), -3.0)
                    transient_smooth = f(row.get("best_transient_smooth"), 0.3)
                    drc_strength = f(row.get("best_drc_strength"), None)
                    if drc_strength is None:
                        drc_strength = f(row.get("best_compressor_strength"), 0.4)

                    print(
                        f"[DSP ] {track} -> {out_path.name} "
                        f"(low={low_gain_db:+.1f}, mid={mid_gain_db:+.1f}, high={high_gain_db:+.1f}, "
                        f"ts={transient_smooth:.2f}, drc={drc_strength:.2f})"
                    )

                    y_proc = apply_dsp(
                        y,
                        sr,
                        vocal_cut=0.0,
                        low_gain_db=low_gain_db,
                        mid_gain_db=mid_gain_db,
                        high_gain_db=high_gain_db,
                        transient_smooth=transient_smooth,
                        drc_strength=drc_strength,
                    )
                    y_proc = np.asarray(y_proc, dtype=np.float32)
                    note, rms_in, rms_out, peak_out, gain_db, did_limit = write_processed(out_path, y_proc, sr, "rendered dsp_grid")

                else:
                    print(f"[WARN] {track}: unknown chosen_type='{chosen_type}', rendering instrumental fallback.")
                    note, rms_in, rms_out, peak_out, gain_db, did_limit = write_processed(out_path, y, sr, "fallback instrumental")

                log_rows.append([
                    track, chosen_type, str(instr_path), str(out_path),
                    "OK", note,
                    f"{rms_in:.3f}", f"{rms_out:.3f}", f"{peak_out:.3f}",
                    f"{gain_db:.3f}", str(bool(did_limit)),
                ])

            except Exception as e:
                print(f"[FAIL] {track}: {e}")
                log_rows.append([track, chosen_type, str(instr_path), str(out_path), "FAIL", str(e), "", "", "", "", ""])

    with open(log_csv, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(log_header)
        writer.writerows(log_rows)

    print(f"\n[INFO] Done. Output folder: {out_dir}")
    print(f"[INFO] Log written to: {log_csv}")


if __name__ == "__main__":
    main()
