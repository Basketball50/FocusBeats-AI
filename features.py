from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import numpy as np
import librosa


PathLike = Union[str, Path]


def _safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def _tempo_bpm(y: np.ndarray, sr: int) -> float:
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    try:
        tempo_arr = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr)
        tempo = float(tempo_arr[0]) if np.size(tempo_arr) else 0.0
    except Exception:
        tempo_arr = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        tempo = float(tempo_arr[0]) if np.size(tempo_arr) else 0.0
    return _safe_float(tempo, 0.0)


def features_for_file(path: PathLike) -> Dict[str, float]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")

    y, sr = librosa.load(str(p), sr=None, mono=True)
    y = np.asarray(y, dtype=np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    dur = float(len(y) / sr) if sr > 0 else 0.0


    mean_abs = float(np.mean(np.abs(y))) if len(y) else 0.0
    std = float(np.std(y)) if len(y) else 0.0

   
    hop = 512
    frame = 2048

    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame, hop_length=hop)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame, hop_length=hop)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=frame, hop_length=hop)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=frame, hop_length=hop)
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=frame, hop_length=hop)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=frame, hop_length=hop)
    mfcc_delta = librosa.feature.delta(mfcc)


    tempo = _tempo_bpm(y, sr)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop)
    onset_rate = float(len(onsets) / dur) if dur > 1e-6 else 0.0

  
    def mean_std(A):
        return float(np.mean(A)), float(np.std(A))

    rms_mean, rms_std = mean_std(rms)
    zcr_mean, zcr_std = mean_std(zcr)
    centroid_mean, centroid_std = mean_std(centroid)
    rolloff_mean, rolloff_std = mean_std(rolloff)
    bandwidth_mean, bandwidth_std = mean_std(bandwidth)
    flatness_mean, flatness_std = mean_std(flatness)

    delta_mag = float(np.mean(np.abs(mfcc_delta))) if mfcc_delta.size else 0.0

    out: Dict[str, float] = {}

    out["duration_sec"] = _safe_float(dur)
    out["sr"] = _safe_float(sr)
    out["mean_abs"] = _safe_float(mean_abs)
    out["std"] = _safe_float(std)

    out["rms_mean"] = _safe_float(rms_mean)
    out["rms_std"] = _safe_float(rms_std)
    out["zcr_mean"] = _safe_float(zcr_mean)
    out["zcr_std"] = _safe_float(zcr_std)

    out["centroid_mean"] = _safe_float(centroid_mean)
    out["centroid_std"] = _safe_float(centroid_std)
    out["rolloff_mean"] = _safe_float(rolloff_mean)
    out["rolloff_std"] = _safe_float(rolloff_std)
    out["bandwidth_mean"] = _safe_float(bandwidth_mean)
    out["bandwidth_std"] = _safe_float(bandwidth_std)
    out["flatness_mean"] = _safe_float(flatness_mean)
    out["flatness_std"] = _safe_float(flatness_std)

    out["tempo_bpm"] = _safe_float(tempo)
    out["onset_rate_per_sec"] = _safe_float(onset_rate)

    for i in range(13):
        m, s = mean_std(mfcc[i : i + 1, :])
        out[f"mfcc{i+1}_mean"] = _safe_float(m)
        out[f"mfcc{i+1}_std"] = _safe_float(s)

    out["rms"] = _safe_float(rms_mean)
    out["centroid"] = _safe_float(centroid_mean)
    out["flatness"] = _safe_float(flatness_mean)
    out["tempo"] = _safe_float(tempo)
    out["onset_rate"] = _safe_float(onset_rate)
    out["delta"] = _safe_float(delta_mag)

    return out
