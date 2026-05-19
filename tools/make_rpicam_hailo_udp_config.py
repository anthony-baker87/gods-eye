from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an rpicam Hailo JSON that emits object detections over UDP.")
    parser.add_argument(
        "--source",
        default="/usr/share/rpi-camera-assets/hailo_yolov8_inference.json",
        help="Base rpicam Hailo post-process JSON.",
    )
    parser.add_argument("--output", default="hailo_yolov8_udp.json", help="Output JSON path.")
    parser.add_argument("--host", default="127.0.0.1", help="UDP host for object_detect_udp.")
    parser.add_argument("--port", type=int, default=12347, help="UDP port for object_detect_udp.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)

    with source.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("rpicam post-process JSON must be an object.")

    data["object_detect_udp"] = {
        "ip": args.host,
        "port": args.port,
    }
    _ensure_stage_order(data)

    output.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    return 0


def _ensure_stage_order(data: dict[str, Any]) -> None:
    """Place UDP after Hailo inference and before drawing when those stages exist."""

    ordered: dict[str, Any] = {}
    udp_stage = data.get("object_detect_udp")
    inserted = False
    for key, value in data.items():
        if key == "object_detect_udp":
            continue
        if key == "object_detect_draw_cv" and udp_stage is not None:
            ordered["object_detect_udp"] = udp_stage
            inserted = True
        ordered[key] = value
    if not inserted and udp_stage is not None:
        ordered["object_detect_udp"] = udp_stage
    data.clear()
    data.update(ordered)


if __name__ == "__main__":
    raise SystemExit(main())
