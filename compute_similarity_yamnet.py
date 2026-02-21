import argparse
from pathlib import Path

import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub

ROOT = Path(__file__).resolve().parents[1]

def load_yamnet():
    print("[INFO] Loading YAMNet from TF Hub...")
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    print("[INFO] YAMNet loaded.")
    return model

def file_embedding(model, path):
    y, sr = librosa.load(path, sr=16000, mono=True)
    if y.size == 0:
        raise ValueError(f"Empty audio file: {path}")
    waveform = tf.convert_to_tensor(y, dtype=tf.float32)
    scores, embeddings, spectrogram = model(waveform)
    emb = tf.reduce_mean(embeddings, axis=0)
    return emb.numpy()

def cosine_similarity(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    num = np.dot(a, b)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den == 0.0:
        return 0.0
    return float(num / den)

def main():
    parser = argparse.ArgumentParser(
        description="Compute YAMNet embedding similarity between two audio files."
    )
    parser.add_argument("original", type=str, help="Path to original WAV")
    parser.add_argument("transformed", type=str, help="Path to transformed WAV")
    args = parser.parse_args()

    orig_path = Path(args.original)
    trans_path = Path(args.transformed)

    if not orig_path.is_absolute():
        orig_path = ROOT / orig_path
    if not trans_path.is_absolute():
        trans_path = ROOT / trans_path

    if not orig_path.exists():
        raise SystemExit(f"Original file not found: {orig_path}")
    if not trans_path.exists():
        raise SystemExit(f"Transformed file not found: {trans_path}")

    print("[INFO] Original   :", orig_path)
    print("[INFO] Transformed:", trans_path)

    yamnet = load_yamnet()

    emb_orig = file_embedding(yamnet, orig_path)
    emb_trans = file_embedding(yamnet, trans_path)

    sim = cosine_similarity(emb_orig, emb_trans)
    dist = 1.0 - sim

    print("\n=== YAMNet Embedding Similarity ===")
    print(f"Cosine similarity : {sim:.4f}")
    print(f"Distance (1 - sim): {dist:.4f}")

if __name__ == "__main__":
    main()
