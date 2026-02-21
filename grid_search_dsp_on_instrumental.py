from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

import numpy as np
import librosa

from scripts.focus_yamnet import load_focus_bundle, cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
GENERAL_DIR = ROOT / "data" / "processed" / "general"
INSTR_DIR = ROOT / "data" / "transformed_instrumental"
EVAL_OUT_CSV = ROOT / "data" / "eval" / "knob_grid_alltracks.csv"
SCRIPTS_DIR = ROOT / "scripts"

LOW_GAIN_DB_LIST = [-3.0, 0.0, +3.0]
MID_GAIN_DB_LIST = [-6.0, -3.0, 0.0]
HIGH_GAIN_DB_LIST = [-6.0, -3.0, 0.0]
TRANSIENT_SMOOTH_LIST = [0.0, 0.3, 0.6]
DRC_STRENGTH_LIST = [0.0, 0.4, 0.8]


def preprocess_audio_16k(path: Path) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=16000, mono=True)
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        raise ValueError(f"Empty audio: {path}")
    return y


def preprocess_wave_to_16k(y: np.ndarray, sr: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    return np.asarray(y, dtype=np.float32)


def import_apply_dsp():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.append(str(SCRIPTS_DIR))
    try:
        from apply_dsp_knobs import apply_dsp
    except ImportError as e:
        raise SystemExit(f"Could not import apply_dsp from apply_dsp_knobs.py: {e}")
    return apply_dsp


def main():
    (ROOT / "data" / "eval").mkdir(parents=True, exist_ok=True)

    wav_files = sorted(GENERAL_DIR.glob("*.wav"))
    if not wav_files:
        print("[WARN] No .wav files found in", GENERAL_DIR)
        return

    print(f"[INFO] Grid-search DSP on ALL general tracks ({len(wav_files)} found). CSV only.")

    focus = load_focus_bundle(ROOT)
    apply_dsp = import_apply_dsp()

    header = [
        "track",
        "low_gain_db",
        "mid_gain_db",
        "high_gain_db",
        "transient_smooth",
        "drc_strength",
        "focusability_orig",
        "focusability_instr",
        "sim_instr",
        "focusability_variant",
        "sim_variant",
        "variant_id",
        "focus_pos_label",
    ]
    rows = []
    skipped_no_instr = 0

    for orig_path in wav_files:
        stem = orig_path.stem
        instr_path = INSTR_DIR / f"{stem}_instrumental.wav"
        if not instr_path.exists():
            skipped_no_instr += 1
            continue

        y_orig_16k = preprocess_audio_16k(orig_path)
        emb_orig = focus.yamnet_mean_embedding(y_orig_16k)
        focus_orig = focus.focusability(emb_orig)

        y_instr_16k = preprocess_audio_16k(instr_path)
        emb_instr = focus.yamnet_mean_embedding(y_instr_16k)
        focus_instr = focus.focusability(emb_instr)
        sim_instr = cosine_similarity(emb_orig, emb_instr)

        y_wave, sr = librosa.load(str(instr_path), sr=None, mono=True)

        for low_db, mid_db, high_db, ts, drc in itertools.product(
            LOW_GAIN_DB_LIST, MID_GAIN_DB_LIST, HIGH_GAIN_DB_LIST, TRANSIENT_SMOOTH_LIST, DRC_STRENGTH_LIST
        ):
            variant_id = f"l{low_db:+.1f}_m{mid_db:+.1f}_h{high_db:+.1f}_ts{ts:.2f}_drc{drc:.2f}"

            y_proc = apply_dsp(
                y_wave, sr,
                vocal_cut=0.0,
                low_gain_db=low_db,
                mid_gain_db=mid_db,
                high_gain_db=high_db,
                transient_smooth=ts,
                drc_strength=drc,
            )

            y_proc_16k = preprocess_wave_to_16k(y_proc, sr)
            emb_var = focus.yamnet_mean_embedding(y_proc_16k)
            focus_var = focus.focusability(emb_var)
            sim_var = cosine_similarity(emb_orig, emb_var)

            rows.append([
                stem, low_db, mid_db, high_db, ts, drc,
                float(focus_orig), float(focus_instr), float(sim_instr),
                float(focus_var), float(sim_var),
                variant_id, focus.pos_label
            ])

    with open(EVAL_OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print("[INFO] Saved:", EVAL_OUT_CSV)
    if skipped_no_instr:
        print(f"[INFO] Skipped {skipped_no_instr} tracks missing instrumental in {INSTR_DIR}.")


if __name__ == "__main__":
    main()
