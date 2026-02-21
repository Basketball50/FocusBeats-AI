from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import librosa


AUDIO_EXTS = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".m4a", ".ogg"}


def ensure_stereo_ch_first(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)

    if y.ndim == 1:
        return np.stack([y, y], axis=0).astype(np.float32)

    if y.ndim != 2:
        y1 = np.asarray(y).reshape(-1).astype(np.float32)
        return np.stack([y1, y1], axis=0).astype(np.float32)

    a, b = y.shape  

    if b == 2 and a != 2:
        return y.T.astype(np.float32)

    if a == 2:
        return y[:2].astype(np.float32)

    if b == 1:
        y1 = y[:, 0].astype(np.float32)
        return np.stack([y1, y1], axis=0).astype(np.float32)

    if b > 2:
        return y[:, :2].T.astype(np.float32)

    return y[:2].astype(np.float32)


def loop_to_length(y2n: np.ndarray, target_n: int) -> np.ndarray:
    n = int(y2n.shape[1])
    if n <= 0:
        return np.zeros((2, target_n), dtype=np.float32)
    reps = (target_n + n - 1) // n
    y = np.tile(y2n, (1, reps))[:, :target_n]
    return y.astype(np.float32)


def peak_limit(y: np.ndarray, peak: float = 0.98) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    p = float(np.max(np.abs(y))) if y.size else 0.0
    if p > peak and p > 0:
        y = y * (peak / (p + 1e-12))
    return y.astype(np.float32)


def rms_db(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float32)
    r = float(np.sqrt(np.mean(y * y) + 1e-12))
    return 20.0 * np.log10(r + 1e-12)


def normalize_rms(y: np.ndarray, target_rms_db: float) -> np.ndarray:
    cur = rms_db(y)
    gain_db = target_rms_db - cur
    gain = 10.0 ** (gain_db / 20.0)
    return (y * gain).astype(np.float32)


def make_intermittent_mask(
    n: int,
    sr: int,
    on_min_s: float,
    on_max_s: float,
    off_min_s: float,
    off_max_s: float,
    start_offset_s: float = 0.0,
    fade_s: float = 0.12,
) -> np.ndarray:
    mask = np.zeros((n,), dtype=np.float32)
    t = int(max(0.0, start_offset_s) * sr)
    fade_n = max(1, int(fade_s * sr))

    while t < n:
        off = int(random.uniform(off_min_s, off_max_s) * sr)
        t += off
        if t >= n:
            break

        on = int(random.uniform(on_min_s, on_max_s) * sr)
        t_end = min(n, t + on)
        seg_len = t_end - t
        if seg_len <= 0:
            t = t_end
            continue

        fn = min(fade_n, seg_len // 2) if seg_len >= 2 else 1

        if fn > 0:
            mask[t : t + fn] = np.linspace(0.0, 1.0, fn, endpoint=False, dtype=np.float32)
     
        s0 = t + fn
        s1 = t_end - fn
        if s1 > s0:
            mask[s0:s1] = 1.0
        
        if fn > 0:
            mask[t_end - fn : t_end] = np.linspace(1.0, 0.0, fn, endpoint=False, dtype=np.float32)

        t = t_end

    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def mix_layer_intermittent(
    base2n: np.ndarray,
    layer2n: np.ndarray,
    gain_db: float,
    mask_n: np.ndarray,
    duck_db: float,
) -> np.ndarray:
    n = int(base2n.shape[1])
    mask = mask_n.reshape(1, n)

    gain = 10.0 ** (gain_db / 20.0)
    duck = 10.0 ** (-abs(duck_db) / 20.0)

    base_ducked = base2n * (1.0 - mask + mask * duck)
    out = base_ducked + (layer2n * gain) * mask
    return out.astype(np.float32)


class CyclingPicker:
    def __init__(self, items: List[Path], rng: random.Random):
        self.items = list(items)
        self.rng = rng
        self.i = 0
        if self.items:
            self.rng.shuffle(self.items)

    def pick(self) -> Optional[Path]:
        if not self.items:
            return None
        if self.i >= len(self.items):
            self.rng.shuffle(self.items)
            self.i = 0
        out = self.items[self.i]
        self.i += 1
        return out


def list_layers(layer_dir: Path) -> List[Path]:
    return sorted([p for p in layer_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS])


def bucket_kind(p: Path) -> str:
    s = str(p).lower()
    if "vinyl" in s or "crackle" in s or "dust" in s:
        return "vinyl"
    if "rain" in s or "ambient" in s or "ambience" in s or "field" in s:
        return "ambient"
    if "pad" in s or "chord" in s or "texture" in s:
        return "pad"
    if "hiss" in s or "noise" in s:
        return "vinyl"
    return "pad"


@dataclass
class LofiParams:
    sr: int = 44100
    max_layers: int = 2

    vinyl_db: float = -27.0
    pad_db: float = -28.0
    ambient_db: float = -30.0

    duck_db: float = 0.6
    duck_vinyl_db: float = 0.0

    on_min_s: float = 5.0
    on_max_s: float = 10.0
    off_min_s: float = 8.0
    off_max_s: float = 18.0
    fade_s: float = 0.12

    target_rms_db: float = -18.0
    peak: float = 0.98


def apply_lofi_layers(
    base2n: np.ndarray,
    sr: int,
    layers_dir: Path,
    amount: float,
    seed: int = 0,
    params: Optional[LofiParams] = None,
) -> Tuple[np.ndarray, List[str]]:
    params = params or LofiParams(sr=sr)
    amt = float(np.clip(amount, 0.0, 1.0))

    if amt <= 0.01:
        return base2n.astype(np.float32), []

    rng = random.Random(seed)

    layer_files = list_layers(layers_dir)
    if not layer_files:
        return base2n.astype(np.float32), []

    buckets: Dict[str, List[Path]] = {"vinyl": [], "ambient": [], "pad": []}
    for p in layer_files:
        buckets[bucket_kind(p)].append(p)

    pickers = {
        "vinyl": CyclingPicker(buckets["vinyl"], random.Random(seed + 101)),
        "pad": CyclingPicker(buckets["pad"], random.Random(seed + 202)),
        "ambient": CyclingPicker(buckets["ambient"], random.Random(seed + 303)),
    }

    y = base2n.astype(np.float32)
    n = int(y.shape[1])

    picks: List[Tuple[str, Path, float]] = []
    v = pickers["vinyl"].pick()
    if v is not None:
        picks.append(("vinyl", v, params.vinyl_db))

    pa: List[Tuple[str, Path, float]] = []
    p = pickers["pad"].pick()
    a = pickers["ambient"].pick()
    if p is not None:
        pa.append(("pad", p, params.pad_db))
    if a is not None:
        pa.append(("ambient", a, params.ambient_db))
    rng.shuffle(pa)

    remaining = max(0, params.max_layers - len(picks))
    picks.extend(pa[:remaining])

    used: List[str] = []

    def scale_gain(db: float) -> float:
        atten = 20.0 * np.log10(max(amt, 1e-3))
        return float(db + 0.5 * atten)

    duck_db = float(params.duck_db * amt)
    duck_vinyl_db = float(params.duck_vinyl_db * amt)

    for kind, layer_path, base_gain_db in picks:
        try:
            y_layer, _ = librosa.load(str(layer_path), sr=sr, mono=False)
            y_layer = ensure_stereo_ch_first(y_layer)
            y_layer = loop_to_length(y_layer, n)

            if kind == "vinyl":
                mask = make_intermittent_mask(
                    n, sr,
                    on_min_s=max(3.0, params.on_min_s * 0.7),
                    on_max_s=max(5.0, params.on_max_s * 0.7),
                    off_min_s=max(6.0, params.off_min_s * 0.7),
                    off_max_s=max(12.0, params.off_max_s * 0.7),
                    start_offset_s=rng.uniform(0.0, 6.0),
                    fade_s=params.fade_s,
                )
                ddb = duck_vinyl_db
            else:
                mask = make_intermittent_mask(
                    n, sr,
                    on_min_s=params.on_min_s,
                    on_max_s=params.on_max_s,
                    off_min_s=params.off_min_s,
                    off_max_s=params.off_max_s,
                    start_offset_s=rng.uniform(0.0, 10.0),
                    fade_s=params.fade_s,
                )
                ddb = duck_db

            gdb = scale_gain(base_gain_db)
            y = mix_layer_intermittent(y, y_layer, gain_db=gdb, mask_n=mask, duck_db=ddb)
            used.append(f"{kind}:{layer_path.name}@{gdb:.1f}dB")

        except Exception:
            continue

    y = normalize_rms(y, params.target_rms_db)
    y = peak_limit(y, params.peak)
    return y.astype(np.float32), used
