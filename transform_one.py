from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import librosa
import soundfile as sf

from scripts.features import features_for_file
from scripts.dsp import apply_dsp

try:
    from scripts.focus_yamnet import (
        load_yamnet,
        load_focus_model,
        compute_yamnet_embedding,
        score_file,
    )
except Exception:
    from scripts.score_focus_yamnet import (
        load_yamnet,
        load_focus_model,
        compute_yamnet_embedding,
        score_file,
    )

from scripts.lofi_layers import apply_lofi_layers, ensure_stereo_ch_first

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTROLLER_MODEL = ROOT / "models" / "xgb_ranker_controller.json"
DEFAULT_CONTROLLER_COLS = ROOT / "models" / "controller_feature_cols.json"
TARGET_RMS_DB = -18.0
PEAK_LIMIT = 0.98
SILENCE_FLOOR_DB = -60.0


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))

def _rms_db(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return -120.0
    r = float(np.sqrt(np.mean(y * y) + 1e-12))
    return float(20.0 * np.log10(r + 1e-12))


def _normalize_rms(y: np.ndarray, target_rms_db: float, silence_floor_db: float = SILENCE_FLOOR_DB) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    cur = _rms_db(y)
    if cur < float(silence_floor_db):
        return y.astype(np.float32)
    gain_db = float(target_rms_db) - cur
    gain = float(10.0 ** (gain_db / 20.0))
    return (y * gain).astype(np.float32)


def _peak_limit(y: np.ndarray, peak: float = PEAK_LIMIT) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y.astype(np.float32)
    p = float(np.max(np.abs(y))) + 1e-12
    if p > float(peak):
        y = y * (float(peak) / p)
    return y.astype(np.float32)


def _normalize_for_metrics_mono(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    y = _normalize_rms(y, TARGET_RMS_DB, silence_floor_db=SILENCE_FLOOR_DB)
    y = _peak_limit(y, PEAK_LIMIT)
    return y.astype(np.float32)


def _normalize_for_output_stereo(y_stereo_n2: np.ndarray) -> np.ndarray:
    y = np.asarray(y_stereo_n2, dtype=np.float32)
    if y.size == 0:
        return y.astype(np.float32)
    if y.ndim != 2 or y.shape[1] < 2:
        mono = y.reshape(-1).astype(np.float32)
        mono = _normalize_for_metrics_mono(mono)
        if mono.ndim == 1:
            mono = mono.reshape(-1, 1)
        if mono.shape[1] == 1:
            mono = np.concatenate([mono, mono], axis=1)
        return mono.astype(np.float32)

    mono = np.mean(y[:, :2], axis=1).astype(np.float32)
    cur = _rms_db(mono)
    if cur >= float(SILENCE_FLOOR_DB):
        gain_db = float(TARGET_RMS_DB) - cur
        gain = float(10.0 ** (gain_db / 20.0))
        y = (y * gain).astype(np.float32)

    p = float(np.max(np.abs(y))) + 1e-12
    if p > float(PEAK_LIMIT):
        y = (y * (float(PEAK_LIMIT) / p)).astype(np.float32)

    return y.astype(np.float32)


def _load_controller_cols(path: Path) -> List[str]:
    cols = json.loads(path.read_text())
    if not isinstance(cols, list) or not cols:
        raise RuntimeError(f"controller_feature_cols.json invalid: {path}")
    return [str(c) for c in cols]


def _ensure_wav_preserve_stereo(in_path: Path) -> Path:
    y, sr = librosa.load(str(in_path), sr=None, mono=False)
    if y is None or (isinstance(y, np.ndarray) and y.size == 0):
        raise RuntimeError(f"Empty/invalid audio: {in_path}")

    if isinstance(y, np.ndarray) and y.ndim == 2:
        y_to_write = y.T.astype(np.float32)
    else:
        y_to_write = np.asarray(y, dtype=np.float32)

    tmp_dir = Path(tempfile.mkdtemp(prefix="fb_"))
    wav_path = tmp_dir / (in_path.stem + "_src.wav")
    sf.write(str(wav_path), y_to_write, sr)
    return wav_path


def _center_cancel_vocal_reduce(stereo_wav: Path) -> Path:
    y, sr = sf.read(str(stereo_wav), always_2d=True)
    if y.shape[1] < 2:
        mono = y[:, 0].astype(np.float32)
    else:
        L = y[:, 0].astype(np.float32)
        R = y[:, 1].astype(np.float32)
        mono = 0.5 * (L - R)

    mx = float(np.max(np.abs(mono))) if mono.size else 0.0
    if mx > 0:
        mono = 0.97 * mono / mx

    out_dir = ROOT / "data" / "instrumental_tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stereo_wav.stem}_center_cancel.wav"
    sf.write(str(out_path), mono, sr)
    return out_path


def _run_demucs_make_instrumental(input_wav: Path) -> Tuple[Path, Dict[str, Any]]:
    demucs_out = ROOT / "data" / "demucs_out"
    demucs_out.mkdir(parents=True, exist_ok=True)

    demucs_cmd = os.getenv("DEMUCS_CMD", "").strip()
    if demucs_cmd:
        base = demucs_cmd.split()
    else:
        base = [sys.executable, "-m", "demucs"]

    cmd = base + ["-n", "htdemucs", "-o", str(demucs_out), str(input_wav)]

    import subprocess
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Demucs failed (needed to remove vocals).\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{(proc.stdout or '')[-2500:]}\n"
            f"STDERR:\n{(proc.stderr or '')[-2500:]}\n"
        )

    stem_dir = demucs_out / "htdemucs" / input_wav.stem
    if not stem_dir.exists():
        raise RuntimeError(f"Demucs output folder not found: {stem_dir}")

    stem_names = ["drums", "bass", "other"]
    stems = []
    sr_ref = None

    for name in stem_names:
        p = stem_dir / f"{name}.wav"
        if not p.exists():
            continue
        y, sr = librosa.load(str(p), sr=None, mono=True)
        if y.size == 0:
            continue
        if sr_ref is None:
            sr_ref = sr
        elif sr != sr_ref:
            y = librosa.resample(y, orig_sr=sr, target_sr=sr_ref)
        stems.append(y.astype(np.float32))

    if not stems or sr_ref is None:
        raise RuntimeError(f"No usable stems found in {stem_dir}")

    min_len = min(len(s) for s in stems)
    stems = [s[:min_len] for s in stems]
    mix = np.zeros(min_len, dtype=np.float32)
    for s in stems:
        mix += s

    mx = float(np.max(np.abs(mix)))
    if mx > 0:
        mix = 0.97 * mix / mx

    out_dir = ROOT / "data" / "instrumental_tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{input_wav.stem}_instrumental.wav"
    sf.write(str(out_path), mix, sr_ref)

    meta = {
        "vocals_removed_method": "demucs_htdemucs",
        "demucs_cmd": " ".join(cmd),
        "demucs_stem_dir": str(stem_dir),
    }
    return out_path, meta


def _maybe_trim(path: Path, trim_seconds: float) -> Path:
    if trim_seconds <= 0:
        return path
    y, sr = librosa.load(str(path), sr=None, mono=True)
    max_len = int(trim_seconds * sr)
    if len(y) <= max_len:
        return path
    y = y[:max_len]
    out_dir = ROOT / "data" / "trim_tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}_trim{int(trim_seconds)}.wav"
    sf.write(str(out_path), y.astype(np.float32), sr)
    return out_path


def _compute_base_features(audio_path: Path) -> Dict[str, float]:
    feats = features_for_file(audio_path)
    out: Dict[str, float] = {}
    for k, v in feats.items():
        out[str(k)] = _safe_float(v, 0.0)

    out["tempo"] = out.get("tempo_bpm", out.get("tempo", 0.0))
    out["rms"] = out.get("rms_mean", out.get("rms", 0.0))
    out["centroid"] = out.get("centroid_mean", out.get("centroid", 0.0))
    out["onset_rate"] = out.get("onset_rate_per_sec", out.get("onset_rate", 0.0))
    out["flatness"] = out.get("flatness_mean", out.get("flatness", 0.0))
    return out


def _load_xgb_model(controller_model_path: Path):
    try:
        import xgboost as xgb
    except Exception as e:
        raise RuntimeError(
            "xgboost is required for controller knob selection.\n"
            "Install:\n"
            "  python3 -m pip install xgboost\n"
            "If on Mac and you see libomp issues:\n"
            "  brew install libomp\n"
            f"Error: {e}"
        )
    booster = xgb.Booster()
    booster.load_model(str(controller_model_path))
    return booster, xgb


def _candidate_grid() -> List[Dict[str, float]]:
    vocal_cut_vals = [0.20, 0.35, 0.50]
    low_gain_vals = [-1.0, 0.0, 2.0]
    mid_gain_vals = [-0.5, -1.5, -3.0]
    high_gain_vals = [-2.0, -4.0, -6.0]
    transient_vals = [0.05, 0.15, 0.30]
    drc_vals = [0.10, 0.25, 0.45]
    lofi_vals = [0.55, 0.70, 0.85]

    out = []
    for vc in vocal_cut_vals:
        for lg in low_gain_vals:
            for mg in mid_gain_vals:
                for hg in high_gain_vals:
                    for ts in transient_vals:
                        for drc in drc_vals:
                            for la in lofi_vals:
                                out.append(
                                    {
                                        "vocal_cut": vc,
                                        "low_gain_db": lg,
                                        "mid_gain_db": mg,
                                        "high_gain_db": hg,
                                        "transient_smooth": ts,
                                        "drc_strength": drc,
                                        "lofi_amount": la,
                                    }
                                )
    return out


def _detect_embedding_columns(cols: List[str]) -> List[str]:
    groups: Dict[str, List[str]] = {}
    for c in cols:
        m = re.match(r"^(.+?)(\d{1,4})$", c)
        if not m:
            continue
        prefix = m.group(1)
        groups.setdefault(prefix, []).append(c)

    if not groups:
        return []

    best = max(groups.items(), key=lambda kv: len(kv[1]))
    if len(best[1]) < 256:
        return []

    prefix = best[0]
    return [
        c
        for c in cols
        if c.startswith(prefix)
        and re.match(r"^" + re.escape(prefix) + r"\d{1,4}$", c)
    ]


def _fill_controller_feature_dict(
    cols: List[str],
    base_feats: Dict[str, float],
    audio_for_embedding: Path,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    feat_dict = dict(base_feats)

    emb_cols = _detect_embedding_columns(cols)
    meta: Dict[str, Any] = {"embedding_cols_detected": len(emb_cols)}

    if emb_cols:
        yamnet = load_yamnet()
        emb = compute_yamnet_embedding(yamnet, audio_for_embedding).astype(np.float32)
        meta["embedding_dim"] = int(emb.shape[0])
        n = min(len(emb_cols), int(emb.shape[0]))
        for i in range(n):
            feat_dict[emb_cols[i]] = float(emb[i])
        meta["embedding_filled"] = int(n)
        meta["embedding_prefix_example"] = emb_cols[0].split("0")[0] if emb_cols else ""
    else:
        meta["embedding_filled"] = 0

    return feat_dict, meta


def _inject_best_features(feat_dict: Dict[str, float], cand: Dict[str, float]) -> None:
    feat_dict["best_low_gain_db"] = float(cand.get("low_gain_db", 0.0))
    feat_dict["best_mid_gain_db"] = float(cand.get("mid_gain_db", 0.0))
    feat_dict["best_high_gain_db"] = float(cand.get("high_gain_db", 0.0))
    feat_dict["best_transient_smooth"] = float(cand.get("transient_smooth", 0.0))
    feat_dict["best_transient_amount"] = float(cand.get("transient_smooth", 0.0))
    feat_dict["best_drc_strength"] = float(cand.get("drc_strength", 0.0))
    feat_dict["best_compressor_strength"] = float(cand.get("drc_strength", 0.0))
    feat_dict.setdefault("best_grid_focus", 0.0)
    feat_dict.setdefault("best_grid_score", 0.0)
    feat_dict.setdefault("best_grid_sim", 0.0)


def _score_candidates_with_ranker(
    booster,
    xgb,
    cols: List[str],
    base_feat_dict: Dict[str, float],
    candidates: List[Dict[str, float]],
) -> Tuple[int, float, Dict[str, Any]]:
    X = np.zeros((len(candidates), len(cols)), dtype=np.float32)

    for i, cand in enumerate(candidates):
        feat_dict = dict(base_feat_dict)
        feat_dict.update(cand)
        _inject_best_features(feat_dict, cand)

        for j, c in enumerate(cols):
            X[i, j] = _safe_float(feat_dict.get(c, 0.0), 0.0)

    dmat = xgb.DMatrix(X, feature_names=cols)
    preds = np.asarray(booster.predict(dmat), dtype=np.float32)

    best_idx = int(np.argmax(preds))
    best_score = float(preds[best_idx])

    missing_cols = []
    for j, c in enumerate(cols):
        if float(np.max(np.abs(X[:, j]))) == 0.0:
            missing_cols.append(c)

    debug = {
        "n_feature_cols": len(cols),
        "missing_features_filled_with_zero": int(len(missing_cols)),
        "missing_feature_examples": missing_cols[:10],
    }
    return best_idx, best_score, debug


def _stable_int_seed(s: str) -> int:
    h = 2166136261
    for b in s.encode("utf-8", errors="ignore"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def _jsonify(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, dict):
        return {str(k): _jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    return x


def _ensure_mono_wav_for_scoring(audio_path: Path) -> Path:
    y, sr = sf.read(str(audio_path), always_2d=True)
    y = y.astype(np.float32)
    if y.shape[1] == 1:
        mono = y[:, 0]
    else:
        mono = np.mean(y[:, :2], axis=1)

    if mono.size < 256:
        raise RuntimeError(f"Audio too short for scoring: {audio_path} (samples={mono.size})")

    mono = _normalize_for_metrics_mono(mono)

    tmp_dir = Path(tempfile.mkdtemp(prefix="fb_score_"))
    out_wav = tmp_dir / (audio_path.stem + "_mono_score.wav")
    sf.write(str(out_wav), mono.astype(np.float32), int(sr))
    return out_wav


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", dest="out", required=True)

    parser.add_argument("--controller_model", default=str(DEFAULT_CONTROLLER_MODEL))
    parser.add_argument("--controller_cols", default=str(DEFAULT_CONTROLLER_COLS))

    parser.add_argument("--trim_seconds", type=float, default=0.0)
    parser.add_argument("--demucs_required", action="store_true")

    parser.add_argument("--lofi_layers_dir", default=str(ROOT / "data" / "lofi_layers_prepped"))
    parser.add_argument("--lofi_enabled", type=int, default=1, help="1=apply lofi (default), 0=skip lofi")
    parser.add_argument("--lofi_gain_db", type=float, default=0.0, help="post-mix lofi gain in dB (default 0.0)")

    parser.add_argument(
        "--write_meta",
        action="store_true",
        help="Write <out>.meta.json next to the output wav for reproducibility.",
    )

    args = parser.parse_args()

    in_path = Path(args.inp)
    out_path = Path(args.out)
    if not in_path.is_absolute():
        in_path = ROOT / in_path
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    controller_model = Path(args.controller_model)
    controller_cols = Path(args.controller_cols)
    lofi_layers_dir = Path(args.lofi_layers_dir)

    if not controller_model.exists():
        raise RuntimeError(f"Missing controller model: {controller_model}")
    if not controller_cols.exists():
        raise RuntimeError(f"Missing controller cols: {controller_cols}")
    if not lofi_layers_dir.exists():
        raise RuntimeError(
            f"Missing lofi layers dir: {lofi_layers_dir}\n"
            f"Expected: {ROOT/'data'/'lofi_layers_prepped'}"
        )

    src_wav = _ensure_wav_preserve_stereo(in_path)

    vocals_meta: Dict[str, Any] = {}
    try:
        instr_wav, vocals_meta = _run_demucs_make_instrumental(src_wav)
    except Exception as e:
        if args.demucs_required:
            raise
        instr_wav = _center_cancel_vocal_reduce(src_wav)
        vocals_meta = {
            "vocals_removed_method": "fallback_center_cancel",
            "demucs_error": str(e)[:1200],
            "note": "Fallback used. For true vocal removal, set DEMUCS_CMD.",
        }

    work_wav = _maybe_trim(instr_wav, float(args.trim_seconds))
    base_feats = _compute_base_features(work_wav)
    cols = _load_controller_cols(controller_cols)
    base_feat_dict, emb_meta = _fill_controller_feature_dict(cols, base_feats, work_wav)

    booster, xgb = _load_xgb_model(controller_model)
    candidates = _candidate_grid()
    best_idx, best_score, debug_meta = _score_candidates_with_ranker(
        booster, xgb, cols, base_feat_dict, candidates
    )
    knobs = dict(candidates[best_idx])
    knobs["controller_score"] = float(best_score)

    dsp_out = out_path.parent / (out_path.stem + "_dsp.wav")
    dsp_out.parent.mkdir(parents=True, exist_ok=True)
    apply_dsp(work_wav, dsp_out, knobs)

    y_dsp, sr_dsp = sf.read(str(dsp_out), always_2d=True)
    y_dsp = y_dsp.astype(np.float32)

    if y_dsp.shape[1] == 1:
        y_base_stereo = np.concatenate([y_dsp, y_dsp], axis=1)
    else:
        y_base_stereo = y_dsp[:, :2]

    base2n = ensure_stereo_ch_first(y_base_stereo)  

    lofi_enabled = int(args.lofi_enabled) != 0
    lofi_gain_db = float(args.lofi_gain_db)

    lofi_amount = float(knobs.get("lofi_amount", 0.70))
    seed = _stable_int_seed(in_path.name)

    if lofi_enabled:
        y_lofi_2n, used_layers = apply_lofi_layers(
            base2n=base2n,
            sr=int(sr_dsp),
            layers_dir=lofi_layers_dir,
            amount=lofi_amount,
            seed=seed,
        )
        if abs(lofi_gain_db) > 1e-9:
            g = _db_to_amp(lofi_gain_db)
            diff = (y_lofi_2n - base2n).astype(np.float32)
            y_lofi_2n = (base2n + diff * g).astype(np.float32)
    else:
        y_lofi_2n = base2n
        used_layers = []

    used_layers = used_layers or []
    lofi_used_layers_str = ";".join([str(s) for s in used_layers])
    lofi_used_layers_count = int(len(used_layers))

    y_final_stereo = np.asarray(y_lofi_2n, dtype=np.float32).T  
    y_final_stereo = _normalize_for_output_stereo(y_final_stereo)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), y_final_stereo.astype(np.float32), int(sr_dsp))

    yamnet = load_yamnet()
    focus_model, mean, scale = load_focus_model()

    raw_score_wav = _ensure_mono_wav_for_scoring(Path(src_wav))       
    instr_score_wav = _ensure_mono_wav_for_scoring(Path(work_wav))     
    dsp_score_wav = _ensure_mono_wav_for_scoring(Path(dsp_out))        
    out_score_wav = _ensure_mono_wav_for_scoring(Path(out_path))     

    raw_focus = float(score_file(yamnet, focus_model, mean, scale, raw_score_wav))
    focus_before = float(score_file(yamnet, focus_model, mean, scale, instr_score_wav))
    focus_post_dsp = float(score_file(yamnet, focus_model, mean, scale, dsp_score_wav))
    focus_after = float(score_file(yamnet, focus_model, mean, scale, out_score_wav))

    emb_before = compute_yamnet_embedding(yamnet, instr_score_wav).astype(np.float32)
    emb_after = compute_yamnet_embedding(yamnet, out_score_wav).astype(np.float32)

    num = float(np.dot(emb_before, emb_after))
    den = float(np.linalg.norm(emb_before) * np.linalg.norm(emb_after))
    sim = float(num / den) if den > 0 else 0.0

    duration_sec = float(y_final_stereo.shape[0] / sr_dsp) if sr_dsp else 0.0

    metrics = {
        "focus_before": focus_before,  
        "focus_after": focus_after,    
        "focus_delta": focus_after - focus_before,
        "yamnet_similarity": sim,
        "yamnet_distance": 1.0 - sim,
        "duration_sec_output": duration_sec,

        "focus_raw": raw_focus,
        "focus_post_dsp": focus_post_dsp,
        "focus_delta_raw_to_final": focus_after - raw_focus,
        "focus_delta_instr_to_post_dsp": focus_post_dsp - focus_before,
        "focus_delta_post_dsp_to_final": focus_after - focus_post_dsp,
    }

    meta = {
        "pipeline": [
            "load/convert -> wav (preserve stereo)",
            "vocals removal -> demucs or fallback center-cancel",
            "optional trim",
            "features_for_file (+ yamnet embedding if needed)",
            "xgb controller selects knobs (best_* injected)",
            "apply dsp knobs (clean EQ + stable compressor)",
            "apply lofi layers (prepped layer files; intermittent textures)",
            "focus score (raw/instrumental/post-dsp/final)",
            "yamnet similarity (instrumental/final)",
        ],
        "input": str(in_path),
        "src_wav": str(src_wav),
        "instrumental_wav": str(work_wav),
        "dsp_wav": str(dsp_out),
        "output": str(out_path),
        "trim_seconds": float(args.trim_seconds),
        "dsp_impl": "scripts.dsp.apply_dsp",
        "controller_model": str(controller_model),
        "controller_cols": str(controller_cols),
        "controller_cols_count": len(cols),

        "lofi_enabled": bool(lofi_enabled),
        "lofi_gain_db": float(lofi_gain_db),

        "lofi_used": bool(lofi_enabled),
        "lofi_amount_used": lofi_amount,
        "lofi_seed": int(seed),
        "lofi_layers_dir": str(lofi_layers_dir),

        "lofi_used_layers": lofi_used_layers_str,
        "lofi_used_layers_count": lofi_used_layers_count,
        "lofi_used_layers_list": _jsonify(used_layers),

        "normalization_target_rms_db": float(TARGET_RMS_DB),
        "normalization_peak_limit": float(PEAK_LIMIT),
        "normalization_silence_floor_db": float(SILENCE_FLOOR_DB),

        **vocals_meta,
        **emb_meta,
        **debug_meta,
    }

    payload = {"meta": meta, "knobs": knobs, "metrics": metrics}

    print(json.dumps(payload, indent=2))

    if args.write_meta:
        meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
        meta_path.write_text(json.dumps(payload, indent=2))
        print(f"[INFO] Wrote metadata: {meta_path}")


if __name__ == "__main__":
    main()
