from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe rpicam-apps Hailo metadata output.")
    parser.add_argument("--post-process-file", default="/usr/share/rpi-camera-assets/hailo_yolov8_inference.json")
    parser.add_argument("--metadata", default="output/hailo_metadata_probe.txt")
    parser.add_argument("--seconds", type=int, default=10)
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if metadata_path.exists():
        metadata_path.unlink()

    command = [
        "rpicam-hello",
        "-t",
        f"{args.seconds}s",
        "--post-process-file",
        args.post_process_file,
        "--lores-width",
        "640",
        "--lores-height",
        "640",
        "--metadata",
        str(metadata_path),
    ]
    print("Running:")
    print(" ".join(command))
    completed = subprocess.run(command, check=False)
    print(f"rpicam exited with code {completed.returncode}")

    if not metadata_path.exists():
        print(f"No metadata file was written: {metadata_path}")
        return 1

    text = metadata_path.read_text(encoding="utf-8", errors="replace")
    print(f"\nMetadata file: {metadata_path}")
    print(f"Bytes: {len(text)}")
    print("\n--- first 4000 chars ---")
    print(text[:4000])
    print("--- end preview ---")
    print(f"\nFinished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
