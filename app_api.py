from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "api_uploads"
OUT_DIR = DATA_DIR / "api_outputs"

DEFAULT_CONTROLLER_MODEL = PROJECT_ROOT / "models" / "xgb_ranker_controller.json"
DEFAULT_CONTROLLER_COLS = PROJECT_ROOT / "models" / "controller_feature_cols_NOTEBOOK_ORDER.json"
DEFAULT_CTRL_LOOKUP_CSV = DATA_DIR / "eval" / "controller_dataset_yamnet_m003.csv"

DEFAULT_LOFI_LAYERS_DIR = DATA_DIR / "lofi_layers_prepped"

ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".m4a", ".ogg", ".aac"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "tfhub_cache").mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str) -> str:
    bad = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    out = name
    for b in bad:
        out = out.replace(b, "_")
    out = re.sub(r"\s+", " ", out).strip()
    return out or "file"


def _json_error(msg: str, code: int = 400, extra: Optional[Dict[str, Any]] = None):
    payload: Dict[str, Any] = {"ok": False, "error": msg}
    if extra:
        payload.update(extra)
    return jsonify(payload), code


def _python_for_pipeline() -> str:
    return (os.environ.get("PYTHON") or os.sys.executable).strip()


def _env_for_subprocess() -> Dict[str, str]:
    env = dict(os.environ)
    env.setdefault("TFHUB_CACHE_DIR", str(DATA_DIR / "tfhub_cache"))
    return env


def _check_demucs_available(py: str, env: Dict[str, str]) -> Tuple[bool, str]:
    try:
        p = subprocess.run(
            [py, "-c", "import demucs; print('ok')"],
            capture_output=True,
            text=True,
            env=env,
        )
        if p.returncode == 0:
            return True, ""
        return False, (p.stderr or p.stdout or "").strip()[-1500:]
    except Exception as e:
        return False, str(e)


def _parse_boolish(x: str, default: bool) -> bool:
    if x is None:
        return default
    s = str(x).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    return default


def _parse_float(x: str, default: float) -> float:
    try:
        if x is None:
            return default
        return float(str(x).strip())
    except Exception:
        return default


def run_transform_one_v2(
    input_path: Path,
    output_path: Path,
    controller_model: Path,
    controller_cols: Path,
    ctrl_lookup_csv: Path,
    lofi_layers_dir: Path,
    trim_seconds: float,
    demucs_required: bool,
    lofi_enabled: bool,
    lofi_gain_db: float,
) -> Tuple[bool, str, Dict[str, Any]]:
    py = _python_for_pipeline()
    env = _env_for_subprocess()

    cmd = [
        py,
        "-m",
        "scripts.transform_onev2",  
        "--in",
        str(input_path),
        "--out",
        str(output_path),
        "--write_meta",
        "--controller_model",
        str(controller_model),
        "--controller_cols",
        str(controller_cols),
        "--ctrl_lookup_csv",
        str(ctrl_lookup_csv),
        "--lofi_layers_dir",
        str(lofi_layers_dir),
        "--trim_seconds",
        str(float(trim_seconds)),
        "--lofi_enabled",
        "1" if lofi_enabled else "0",
        "--lofi_gain_db",
        str(float(lofi_gain_db)),
    ]
    if demucs_required:
        cmd.append("--demucs_required")

    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        env=env,
    )

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    debug = {
        "python": py,
        "cmd": " ".join(cmd),
        "cwd": str(PROJECT_ROOT),
        "returncode": proc.returncode,
        "TFHUB_CACHE_DIR": env.get("TFHUB_CACHE_DIR", ""),
        "DEMUCS_CMD": env.get("DEMUCS_CMD", ""),
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
    }

    if proc.returncode != 0:
        err = (stderr or stdout or "transform_onev2 failed").strip()
        return False, err[:4000], debug

    payload = None
    try:
        start = stdout.find("{")
        end = stdout.rfind("}")
        if start != -1 and end != -1 and end > start:
            payload = json.loads(stdout[start : end + 1])
    except Exception:
        payload = None

    debug["parsed_payload"] = bool(payload)
    debug["payload"] = payload if isinstance(payload, dict) else None

    return True, "", debug


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.get("/api/health")
def health():
    return jsonify(
        {
            "ok": True,
            "service": "focusbeats-api",
            "project_root": str(PROJECT_ROOT),
            "python_used_if_PYTHON_unset": os.sys.executable,
            "python_env_override": os.environ.get("PYTHON", ""),
            "TFHUB_CACHE_DIR": os.environ.get("TFHUB_CACHE_DIR", str(DATA_DIR / "tfhub_cache")),
            "DEMUCS_CMD": os.environ.get("DEMUCS_CMD", ""),
            "upload_dir": str(UPLOAD_DIR),
            "out_dir": str(OUT_DIR),
            "defaults": {
                "controller_model": str(DEFAULT_CONTROLLER_MODEL),
                "controller_cols": str(DEFAULT_CONTROLLER_COLS),
                "ctrl_lookup_csv": str(DEFAULT_CTRL_LOOKUP_CSV),
                "lofi_layers_dir": str(DEFAULT_LOFI_LAYERS_DIR),
                "pipeline_module": "scripts.transform_onev2",
            },
        }
    )


@app.post("/api/pipeline")
def pipeline():
    if "file" not in request.files:
        return _json_error("Missing form-data field: file", 400)

    f = request.files["file"]
    if not f.filename:
        return _json_error("Empty filename", 400)

    orig_name = sanitize_filename(Path(f.filename).name)
    ext = Path(orig_name).suffix.lower()
    if ext and ext not in ALLOWED_AUDIO_EXTS:
        return _json_error(f"Unsupported extension: {ext}", 400, {"allowed": sorted(ALLOWED_AUDIO_EXTS)})

    lofi_enabled = _parse_boolish(request.form.get("lofi_enabled"), True)
    lofi_gain_db = _parse_float(request.form.get("lofi_gain_db"), 0.0)

    file_id = str(uuid.uuid4())[:12]
    if not ext:
        ext = ".wav"

    saved_name = f"{file_id}__{Path(orig_name).stem}{ext}"
    input_path = (UPLOAD_DIR / saved_name).resolve()
    f.save(str(input_path))

    for label, p in [
        ("controller_model", DEFAULT_CONTROLLER_MODEL),
        ("controller_cols", DEFAULT_CONTROLLER_COLS),
        ("ctrl_lookup_csv", DEFAULT_CTRL_LOOKUP_CSV),
        ("lofi_layers_dir", DEFAULT_LOFI_LAYERS_DIR),
    ]:
        if not Path(p).exists():
            return _json_error(f"{label} not found", 500, {label: str(p)})

    py = _python_for_pipeline()
    env = _env_for_subprocess()
    ok_demucs, demucs_err = _check_demucs_available(py, env)
    if not ok_demucs and not env.get("DEMUCS_CMD"):
        return _json_error(
            "Demucs is not available in the Python used by the API. "
            "Set PYTHON to the demucs environment python, or set DEMUCS_CMD.",
            500,
            {
                "python_used": py,
                "demucs_import_error": demucs_err,
                "how_to_fix": "Start the API with: export PYTHON=/path/to/demucs_env/bin/python",
            },
        )

    out_base = sanitize_filename(Path(input_path.name).stem)
    job_id = str(uuid.uuid4())[:10]
    out_wav = (OUT_DIR / f"{out_base}__{job_id}.wav").resolve()
    out_meta = Path(str(out_wav) + ".meta.json")

    trim_seconds = 0.0
    demucs_required = True

    ok, err, debug = run_transform_one_v2(
        input_path=input_path,
        output_path=out_wav,
        controller_model=DEFAULT_CONTROLLER_MODEL,
        controller_cols=DEFAULT_CONTROLLER_COLS,
        ctrl_lookup_csv=DEFAULT_CTRL_LOOKUP_CSV,
        lofi_layers_dir=DEFAULT_LOFI_LAYERS_DIR,
        trim_seconds=trim_seconds,
        demucs_required=demucs_required,
        lofi_enabled=lofi_enabled,
        lofi_gain_db=lofi_gain_db,
    )

    if not ok:
        print("\n[PIPELINE ERROR] transform_onev2 failed")
        print(debug.get("cmd"))
        print(debug.get("stderr_tail") or debug.get("stdout_tail"))
        print()
        return _json_error("transform failed", 500, {"details": err, "debug": debug})

    if not out_wav.exists():
        return _json_error(
            "transform succeeded but output wav missing",
            500,
            {"output_wav": str(out_wav), "debug": debug},
        )

    payload = (debug.get("payload") or {}) if isinstance(debug, dict) else {}

    if not out_meta.exists() and isinstance(payload, dict) and payload:
        try:
            out_meta.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    rel_wav = str(out_wav.relative_to(PROJECT_ROOT))
    rel_meta = str(out_meta.relative_to(PROJECT_ROOT)) if out_meta.exists() else ""

    return jsonify(
        {
            "ok": True,
            "job_id": job_id,
            "file_id": file_id,
            "input_path": str(input_path.relative_to(PROJECT_ROOT)),
            "output_wav": rel_wav,
            "output_meta": rel_meta,
            "payload": payload,
            "download_url_wav": f"/api/download?path={rel_wav}",
            "download_url_meta": f"/api/download?path={rel_meta}" if rel_meta else "",
        }
    )


@app.get("/api/download")
def download():
    rel = str(request.args.get("path") or "").strip()
    if not rel:
        return _json_error("Missing query param: path", 400)

    allowed_roots = [
        OUT_DIR,
        UPLOAD_DIR,
        (DATA_DIR / "transformed_selected"),
    ]

    resolved: Optional[Path] = None
    for base in allowed_roots:
        try:
            cand = (PROJECT_ROOT / rel).resolve()
            base_r = base.resolve()
            if str(cand).startswith(str(base_r) + os.sep) and cand.exists():
                resolved = cand
                break
        except Exception:
            continue

    if resolved is None:
        return _json_error("File not found or not allowed", 404, {"path": rel})

    return send_file(str(resolved), as_attachment=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
