import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import resampy
import tensorflow as tf
import tensorflow_hub as hub

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "focus_yamnet_tf.keras"
SCALER_PATH = MODEL_DIR / "focus_yamnet_scaler.npy"

def load_yamnet():
    print("[INFO] Loading YAMNet from TF Hub...")
    handle = "https://tfhub.dev/google/yamnet/1"
    yamnet = hub.load(handle)
    print("[INFO] YAMNet loaded.")
    return yamnet

def load_focus_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Missing scaler file: {SCALER_PATH}")

    model = tf.keras.models.load_model(str(MODEL_PATH))

    scaler_dict = np.load(SCALER_PATH, allow_pickle=True).item()
    mean = scaler_dict["mean"]
    scale = scaler_dict["scale"]

    return model, mean, scale

def compute_yamnet_embedding(yamnet, wav_path: Path):
    waveform, sr = sf.read(str(wav_path))

    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)

    target_sr = 16000
    if sr != target_sr:
        waveform = resampy.resample(waveform, sr, target_sr)

    waveform = waveform.astype(np.float32)

    scores, embeddings, spectrogram = yamnet(waveform)
    embeddings = embeddings.numpy()

    clip_embedding = embeddings.mean(axis=0)

    return clip_embedding

def score_file(yamnet, model, mean, scale, wav_path: Path):
    emb = compute_yamnet_embedding(yamnet, wav_path)

    emb_scaled = (emb - mean) / scale
    emb_scaled = emb_scaled.astype(np.float32)[np.newaxis, :]

    prob = float(model.predict(emb_scaled, verbose=0)[0, 0])
    return prob

def main():
    parser = argparse.ArgumentParser(
        description="Compute focusability score using YAMNet + MLP."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more .wav files to score."
    )
    args = parser.parse_args()

    yamnet = load_yamnet()
    model, mean, scale = load_focus_model()

    for p in args.paths:
        wav_path = Path(p)
        if not wav_path.exists():
            print(f"[WARN] File not found: {wav_path}")
            continue

        score = score_file(yamnet, model, mean, scale, wav_path)
        print(f"{wav_path} -> focusability score: {score:.6f}")


if __name__ == "__main__":
    main()
