from pathlib import Path

import numpy as np
import librosa

PROCESSED_DIR = Path("data/processed")
FEATURES_DIR = Path("data/features")

TARGET_SR = 44100  


def extract_features_for_file(audio_path: Path):
    """
    Load a wav file and compute a set of track-level features.
    Returns (feature_vector, feature_names, duration_sec).
    """
    y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    duration_sec = librosa.get_duration(y=y, sr=sr)

    if len(y) == 0:
        raise ValueError("Empty audio signal")

    n_fft = 2048
    hop_length = 512  

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr
    )

    tempo_confidence = float(len(beat_frames)) / (duration_sec + 1e-6)

    tempo_over_time = librosa.beat.tempo(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        aggregate=None,
    )
    if tempo_over_time is not None and tempo_over_time.size > 1:
        tempo_stability = float(np.std(tempo_over_time))
    else:
        tempo_stability = 0.0  

    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_times = librosa.frames_to_time(onsets, sr=sr)
    onset_rate = len(onset_times) / max(duration_sec, 1e-6)

    if duration_sec > 0 and len(onset_times) > 0:
        n_bins = max(int(np.ceil(duration_sec)), 1)
        per_sec_counts = np.zeros(n_bins, dtype=np.float32)
        for t in onset_times:
            idx = int(np.floor(t))
            if 0 <= idx < n_bins:
                per_sec_counts[idx] += 1
        onset_density_var = float(np.var(per_sec_counts))
    else:
        onset_density_var = 0.0

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

    spec_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    spec_rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]
    spec_flux = librosa.onset.onset_strength(
        S=librosa.amplitude_to_db(S), sr=sr
    )
    spec_flatness = librosa.feature.spectral_flatness(S=S)[0]

    spec_centroid_mean = float(np.mean(spec_centroid))
    spec_centroid_std = float(np.std(spec_centroid))
    spec_rolloff_mean = float(np.mean(spec_rolloff))
    spec_flux_mean = float(np.mean(spec_flux))
    spec_flux_std = float(np.std(spec_flux))
    spec_flatness_mean = float(np.mean(spec_flatness))

    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    total_energy = np.sum(S ** 2)

    if total_energy <= 0:
        high_band_ratio = 0.0
        vocal_band_energy_ratio = 0.0
    else:
        high_band_mask = (freqs >= 2000) & (freqs <= 5000)
        high_energy = np.sum((S[high_band_mask, :]) ** 2)
        high_band_ratio = float(high_energy / total_energy)

        vocal_mask = (freqs >= 300) & (freqs <= 3400)
        vocal_energy = np.sum((S[vocal_mask, :]) ** 2)
        vocal_band_energy_ratio = float(vocal_energy / total_energy)

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length
    )  

    mfcc_means = np.mean(mfcc, axis=1)  
    mfcc_stds = np.std(mfcc, axis=1)   

    mfcc_var_per_coeff = np.var(mfcc, axis=1)  
    mfcc_var_mean = float(np.mean(mfcc_var_per_coeff))
    mfcc_var_std = float(np.std(mfcc_var_per_coeff))

    base_values = np.array(
        [
            float(duration_sec),
            float(tempo),
            float(tempo_confidence),
            float(tempo_stability),
            float(onset_rate),
            float(onset_density_var),
            float(spec_centroid_mean),
            float(spec_centroid_std),
            float(spec_rolloff_mean),
            float(spec_flux_mean),
            float(spec_flux_std),
            float(spec_flatness_mean),
            float(rms_mean),
            float(rms_std),
            float(high_band_ratio),
            float(vocal_band_energy_ratio),
            float(mfcc_var_mean),
            float(mfcc_var_std),
        ],
        dtype=np.float32,
    )

    feature_names = [
        "duration_sec",
        "tempo_bpm",
        "tempo_confidence",
        "tempo_stability_std",
        "onset_rate",
        "onset_density_var",
        "spec_centroid_mean",
        "spec_centroid_std",
        "spec_rolloff_mean",
        "spec_flux_mean",
        "spec_flux_std",
        "spec_flatness_mean",
        "rms_mean",
        "rms_std",
        "high_band_ratio",
        "vocal_band_energy_ratio",
        "mfcc_temporal_var_mean",
        "mfcc_temporal_var_std",
    ]

   
    mfcc_means = mfcc_means.astype(np.float32).ravel()
    mfcc_stds = mfcc_stds.astype(np.float32).ravel()

    feature_vector = np.hstack([base_values, mfcc_means, mfcc_stds])


    for i in range(mfcc_means.shape[0]):
        feature_names.append(f"mfcc_mean_{i}")
    for i in range(mfcc_stds.shape[0]):
        feature_names.append(f"mfcc_std_{i}")

    return feature_vector, feature_names, duration_sec


def process_split(split: str):
    """
    split: "study" or "general"
    """
    in_dir = PROCESSED_DIR / split
    out_dir = FEATURES_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    label = 1 if split == "study" else 0

    for audio_path in sorted(in_dir.iterdir()):
        if not audio_path.is_file():
            continue
        if audio_path.suffix.lower() != ".wav":
            continue

        print(f"[{split}] Extracting features for: {audio_path.name}")
        try:
            feature_vector, feature_names, duration_sec = extract_features_for_file(audio_path)
        except Exception as e:
            print(f"  FAILED on {audio_path}: {e}")
            continue

        out_path = out_dir / (audio_path.stem + ".npz")
        np.savez(
            out_path,
            features=feature_vector,
            feature_names=np.array(feature_names),
            label=label,
            path=str(audio_path),
            duration_sec=duration_sec,
        )


def main():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    for split in ["study", "general"]:
        process_split(split)


if __name__ == "__main__":
    main()
