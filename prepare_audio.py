import subprocess
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
LOG_PATH = Path("logs/conversion_errors.txt")

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

TARGET_SR = 44100     
TARGET_SECONDS = 120   
TARGET_LUFS = -16      


def iter_audio_files(split: str):
    in_root = RAW_DIR / split
    out_root = PROCESSED_DIR / split

    for path in in_root.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in AUDIO_EXTS:
            continue

        out_name = path.stem + ".wav"
        out_path = out_root / out_name
        yield path, out_path


def normalize_and_convert(in_path: Path, out_path: Path):
    loudnorm_filter = f"loudnorm=I={TARGET_LUFS}:TP=-1.5:LRA=11"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error", 
        "-y",                  
        "-i", str(in_path),    
        "-ac", "1",            
        "-ar", str(TARGET_SR), 
        "-af", loudnorm_filter,
        "-t", str(TARGET_SECONDS),  
        "-vn", "-sn", "-dn",  
        str(out_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return False, result.stderr.strip()
    return True, ""


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.joinpath("study").mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.joinpath("general").mkdir(parents=True, exist_ok=True)

    with LOG_PATH.open("w") as log_file:
        for split in ["study", "general"]:
            print(f"Processing split: {split}")
            for in_path, out_path in iter_audio_files(split):
                print(f"  -> {in_path.name} -> {out_path.name}")
                out_path.parent.mkdir(parents=True, exist_ok=True)

                ok, err = normalize_and_convert(in_path, out_path)
                if not ok:
                    msg = f"[{split}] FAILED: {in_path} -> {out_path}\n  Error: {err}\n"
                    print(msg)
                    log_file.write(msg)


if __name__ == "__main__":
    main()
