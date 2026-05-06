#!/usr/bin/env python3
"""
Split WAV files longer than CHUNK_SECONDS into fixed-length chunks using ffmpeg.

Each chunk is saved alongside the original file as:
  original_name_part001.wav, original_name_part002.wav, …

The original file is NOT deleted.

Requirements:
  ffmpeg must be installed and available on PATH.
  (https://ffmpeg.org/download.html)
"""

import json
import subprocess
from pathlib import Path

# ── Presets ───────────────────────────────────────────────────────────────────
SEARCH_DIR    = "./train_diffusion/"  # Root directory to scan
CHUNK_SECONDS = 20                     # Maximum chunk length in seconds
# ─────────────────────────────────────────────────────────────────────────────


def get_duration(path: Path) -> float:
    """Return duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(result.stdout)
    for stream in info.get("streams", []):
        if "duration" in stream:
            return float(stream["duration"])
    raise ValueError(f"Could not determine duration of '{path}'")


def split_wav(path: Path, chunk_secs: int) -> list[Path]:
    """
    Split *path* into chunks of *chunk_secs* seconds with ffmpeg.
    Returns the list of created chunk paths.
    """
    # ffmpeg segment muxer writes files matching the pattern
    out_pattern = path.with_name(f"{path.stem}_part%03d{path.suffix}")

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(path),
            "-f", "segment",
            "-segment_time", str(chunk_secs),
            "-c", "copy",          # stream-copy — no re-encoding, lossless & fast
            "-reset_timestamps", "1",
            str(out_pattern),
        ],
        check=True,
        capture_output=True,
    )

    # Collect and return the files that were actually written
    return sorted(path.parent.glob(f"{path.stem}_part[0-9][0-9][0-9]{path.suffix}"))


def main():
    root = Path(SEARCH_DIR).expanduser().resolve()

    if not root.is_dir():
        print(f"Error: '{root}' is not a valid directory.")
        return

    wav_files = sorted(root.rglob("*.wav"))

    if not wav_files:
        print(f"No WAV files found under '{root}'.")
        return

    print(f"Scanning {len(wav_files)} WAV file(s) under '{root}' …\n")

    split_count = 0
    skip_count  = 0

    for wav in wav_files:
        # Skip files that are themselves chunks produced by this script
        if wav.stem[-8:-3] == "_part":
            continue

        rel = wav.relative_to(root)

        try:
            duration = get_duration(wav)
        except Exception as exc:
            print(f"  ✗ Could not read '{rel}': {exc}")
            continue

        if duration <= CHUNK_SECONDS:
            print(f"  – {rel}  ({duration:.1f}s) — skipped, within limit")
            skip_count += 1
            continue

        print(f"  ↓ {rel}  ({duration:.1f}s) — splitting …")
        try:
            chunks = split_wav(wav, CHUNK_SECONDS)
            for chunk in chunks:
                print(f"      → {chunk.relative_to(root)}")
            split_count += 1
        except subprocess.CalledProcessError as exc:
            print(f"  ✗ ffmpeg failed for '{rel}':\n{exc.stderr.decode()}")

    print(f"\nDone. {split_count} file(s) split, {skip_count} skipped.")


if __name__ == "__main__":
    main()
