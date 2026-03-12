import numpy as np
from pathlib import Path
import librosa
import tensorflow as tf
import tensorflow_hub as hub

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_STUDY = ROOT / "data" / "processed" / "study"
PROCESSED_GENERAL = ROOT / "data" / "processed" / "general"
OUT_PATH = ROOT / "data" / "yamnet_embeddings.npz"

TARGET_SR = 16000

print("[INFO] Loading YAMNet from TF Hub...")
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
print("[INFO] YAMNet loaded.")

def compute_yamnet_embedding(path: Path) -> np.ndarray:
    y, sr = librosa.load(path.as_posix(), sr=TARGET_SR, mono=True)
    waveform = tf.convert_to_tensor(y, dtype=tf.float32)
    scores, embeddings, spectrogram = yamnet(waveform)
    emb_mean = tf.reduce_mean(embeddings, axis=0)
    return emb_mean.numpy()

X = []
y = []
paths = []

study_files = sorted(PROCESSED_STUDY.glob("*.wav"))
general_files = sorted(PROCESSED_GENERAL.glob("*.wav"))

print(f"[INFO] Found {len(study_files)} study tracks, {len(general_files)} general tracks.")

for i, p in enumerate(study_files, 1):
    print(f"[study] {i}/{len(study_files)}  {p.name}")
    emb = compute_yamnet_embedding(p)
    X.append(emb)
    y.append(1)
    paths.append(str(p))

for i, p in enumerate(general_files, 1):
    print(f"[general] {i}/{len(general_files)}  {p.name}")
    emb = compute_yamnet_embedding(p)
    X.append(emb)
    y.append(0)
    paths.append(str(p))

X = np.stack(X)
y = np.array(y)
paths = np.array(paths)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
np.savez(OUT_PATH, X=X, y=y, paths=paths)

print("[INFO] Saved embeddings to:", OUT_PATH)
print("[INFO] X shape:", X.shape)
print("[INFO] y shape:", y.shape)
