from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import librosa
import soundfile as sf
import numpy as np

from scripts.apply_dsp_knobs import apply_dsp as apply_dsp_waveform


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def apply_dsp(input_path: str | Path, output_path: str | Path, knobs: Dict[str, Any]) -> Dict[str, Any]:
    inp = Path(input_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    y, sr = librosa.load(str(inp), sr=None, mono=True)
    y = y.astype(np.float32)

    vocal_cut = _clamp(float(knobs.get("vocal_cut", 0.35)), 0.0, 1.0)

    low_gain_db = _clamp(float(knobs.get("low_gain_db", 0.0)), -6.0, 6.0)
    mid_gain_db = _clamp(float(knobs.get("mid_gain_db", -1.0)), -10.0, 3.0)
    high_gain_db = _clamp(float(knobs.get("high_gain_db", -3.0)), -12.0, 3.0)

    transient_smooth = _clamp(float(knobs.get("transient_smooth", 0.15)), 0.0, 1.0)
    drc_strength = _clamp(float(knobs.get("drc_strength", 0.25)), 0.0, 1.0)

    y_out = apply_dsp_waveform(
        y,
        sr,
        vocal_cut=vocal_cut,
        low_gain_db=low_gain_db,
        mid_gain_db=mid_gain_db,
        high_gain_db=high_gain_db,
        transient_smooth=transient_smooth,
        drc_strength=drc_strength,
    )

    sf.write(str(out), y_out.astype(np.float32), sr)

    return {
        "sr": int(sr),
        "duration_sec": float(len(y_out) / sr) if sr else 0.0,
    }
