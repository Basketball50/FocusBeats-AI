from __future__ import annotations

import numpy as np
import librosa


def _db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _stable_compressor(y: np.ndarray, sr: int, strength: float) -> np.ndarray:
    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 0.0:
        return y

    ratio = 1.0 + 3.5 * s          
    thresh_db = -22.0 + 10.0 * (1.0 - s)  
    thresh = _db_to_lin(thresh_db)

    attack = 0.003 + 0.020 * (1.0 - s)   
    release = 0.050 + 0.200 * s         
    a_a = np.exp(-1.0 / max(1, int(sr * attack)))
    a_r = np.exp(-1.0 / max(1, int(sr * release)))

    x = np.abs(y).astype(np.float32)
    env = np.zeros_like(x, dtype=np.float32)

    for i in range(len(x)):
        if i == 0:
            env[i] = x[i]
            continue
        if x[i] > env[i - 1]:
            env[i] = a_a * env[i - 1] + (1 - a_a) * x[i]
        else:
            env[i] = a_r * env[i - 1] + (1 - a_r) * x[i]

    gain = np.ones_like(env, dtype=np.float32)
    above = env > thresh
    gain[above] = (env[above] / thresh) ** (1.0 / ratio - 1.0)

    y_out = (y * gain).astype(np.float32)

    mx = float(np.max(np.abs(y_out))) if y_out.size else 0.0
    if mx > 0:
        y_out = (0.97 * y_out / mx).astype(np.float32)
    return y_out


def apply_dsp(
    y: np.ndarray,
    sr: int,
    vocal_cut: float = 0.35,
    low_gain_db: float = 0.0,
    mid_gain_db: float = -1.0,
    high_gain_db: float = -3.0,
    transient_smooth: float = 0.15,
    drc_strength: float = 0.25,
) -> np.ndarray:

    if not isinstance(y, np.ndarray):
        y = np.asarray(y, dtype=np.float32)
    if y.ndim != 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    y = y.astype(np.float32)
    if y.size == 0:
        return y

    n_fft = 2048
    hop_length = 512

    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(D).astype(np.float32)
    phase = np.angle(D).astype(np.float32)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    low_band = freqs < 200.0
    mid_band = (freqs >= 200.0) & (freqs <= 4000.0)
    high_band = freqs > 4000.0

    gains = np.ones_like(freqs, dtype=np.float32)

    gains[low_band] *= _db_to_lin(float(low_gain_db))
    gains[mid_band] *= _db_to_lin(float(mid_gain_db))
    gains[high_band] *= _db_to_lin(float(high_gain_db))

    vc = float(np.clip(vocal_cut, 0.0, 1.0))
    vocal_factor = 1.0 - 0.50 * vc 
    gains[mid_band] *= vocal_factor

    ts = float(np.clip(transient_smooth, 0.0, 1.0))
    if ts > 0.0:
        gains[high_band] *= _db_to_lin(-5.0 * ts)

    mag = mag * gains[:, None]

    D_proc = mag * np.exp(1j * phase)
    y_eq = librosa.istft(D_proc, hop_length=hop_length, length=len(y)).astype(np.float32)

    y_out = _stable_compressor(y_eq, sr, float(drc_strength))

    mx = float(np.max(np.abs(y_out))) if y_out.size else 0.0
    if mx > 0:
        y_out = (0.97 * y_out / mx).astype(np.float32)

    return y_out
