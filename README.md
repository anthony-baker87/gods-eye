# God's Eye

Python prototype for onboard person detection and tracking on a Raspberry Pi 5 16GB with the Raspberry Pi AI Kit / Hailo-8L accelerator and an IMX708 wide camera. The app captures frames with Picamera2 or `rpicam-vid`, detects people, assigns persistent track IDs, draws overlays, logs track events, and can serve a local MJPEG dashboard.

The first version is designed to run even away from the hardware by using the `mock` backend and synthetic camera frames. That makes development, dashboard work, and tracker testing possible on a laptop or non-camera Pi.

## Hardware

- Raspberry Pi 5 16GB
- Raspberry Pi AI Kit / Hailo-8L
- Arducam or Raspberry Pi Camera Module 3 Wide IMX708
- Raspberry Pi OS 64-bit
- Adequate cooling and power for sustained camera plus accelerator workloads

## Project Layout

```text
gods-eye/
  README.md
  requirements.txt
  config.yaml
  pyproject.toml
  .github/
    workflows/
      tests.yml
  src/
    main.py
    camera.py
    detector/
      base.py
      hailo_detector.py
      mock_detector.py
    tracking/
      centroid_tracker.py
    dashboard/
      server.py
    gps.py
    overlay.py
    logger.py
    config.py
    utils.py
  tests/
    test_tracker.py
    test_config.py
```

## Raspberry Pi Setup

Update the Pi and enable the camera stack:

```bash
sudo apt update
sudo apt full-upgrade
sudo apt install -y python3-pip python3-venv python3-picamera2 rpicam-apps python3-opencv
```

Check that `rpicam` sees the IMX708 camera:

```bash
rpicam-hello --list-cameras
rpicam-hello -t 5000
```

On older Raspberry Pi OS images, these commands may still be named `libcamera-hello`.

Install the Raspberry Pi AI Kit / Hailo software using Raspberry Pi and Hailo's current official instructions for your OS image. Hailo's older `hailo-rpi5-examples` repository now points developers toward the newer `hailo-apps` infrastructure, while still documenting basic detection pipelines such as `basic_pipelines/detection.py --input rpi`. The exact Hailo Python APIs and example pipelines change over time, so this repo keeps the hardware adapter isolated in `src/detector/hailo_detector.py`.

## Install Python Dependencies

From this directory:

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install Flask PyYAML pytest
```

On Raspberry Pi OS, prefer the system `python3-picamera2`, `python3-opencv`, and `numpy` packages for camera/OpenCV integration. The `--system-site-packages` virtualenv lets the app use those apt packages. On non-Pi development machines, `pip install -r requirements.txt` is fine.

For Pi-specific settings, copy the tracked config and keep the copy local:

```bash
cp config.yaml config.pi.yaml
echo "config.pi*.yaml" >> .git/info/exclude
```

Example Pi camera settings:

```yaml
camera:
  width: 1280
  height: 720
  fps: 30
  source: rpicam
```

## Run With Mock Backend

Mock mode is the safest first run. It uses the configured camera source when available and generates fake moving person detections. If real camera startup fails in mock mode, it can fall back to synthetic frames:

```bash
python -m src.main --config config.pi.yaml --backend mock --debug
```

Open the dashboard from another device on the same network:

```text
http://<pi-ip-address>:8080
```

Or disable the dashboard:

```bash
python -m src.main --config config.pi.yaml --backend mock --no-dashboard
```

## Run With CPU Backend

The CPU backend uses OpenCV HOG plus face detection. It is included as a development fallback, not as the target real-time path:

```bash
python -m src.main --config config.pi.yaml --backend cpu --debug
```

Expect lower accuracy and performance than a modern Hailo model. It works best with good lighting and may detect faces when a full standing person is not visible. The CPU backend downsizes frames internally for inference, so you can use a larger camera stream such as 1280x720 for the dashboard without making CPU detection scale linearly with video size.

## Run With Hailo Backend

Set the backend and model path in `config.yaml`:

```yaml
detection:
  backend: hailo
  confidence_threshold: 0.45
  hailo:
    model_path: /path/to/model.hef
```

Then run:

```bash
python -m src.main --config config.yaml --backend hailo
```

The current `HailoDetector` is an adapter boundary that verifies the Hailo runtime is importable and validates the model path. Wire its `detect()` method to the post-processing used by your selected Raspberry Pi Hailo example, such as a YOLO person detector pipeline. The rest of the app already expects normalized `Detection` objects, so the Hailo-specific code stays contained.

## Configuration

`config.yaml` controls:

- Camera width, height, and FPS
- Camera source: `auto`, `picamera2`, `rpicam`, or `synthetic`
- Detector backend: `hailo`, `cpu`, or `mock`
- Confidence threshold
- Tracker maximum lost frames and matching distance
- Dashboard host, port, and JPEG quality
- Optional GPS provider for detection map pins: disabled, static test coordinates, or `gpsd`
- JSONL logging enabled/path
- Optional annotated video recording path

CLI overrides:

```bash
python -m src.main --config config.yaml --backend mock --record-output --debug
```

## Dashboard Endpoints

- `/` shows MJPEG video with overlays and live status JSON.
- The dashboard map places a "last seen" pin when a human is detected and GPS is enabled.
- `/video.mjpg` streams annotated frames.
- `/status.json` returns current status:

```json
{
  "fps": 29.8,
  "frame_size": [1280, 720],
  "active_track_count": 2,
  "current_detections": [],
  "gps": null,
  "detection_pins": [],
  "uptime": 12.4,
  "detector_backend": "mock"
}
```

## GPS Map Pins

GPS is disabled by default. For bench testing, set a static location in a local, untracked Pi config file:

```yaml
gps:
  enabled: true
  provider: static
  latitude: 34.0522
  longitude: -118.2437
  altitude_m: null
```

For a real GPS receiver exposed through `gpsd`, use:

```yaml
gps:
  enabled: true
  provider: gpsd
  gpsd_host: 127.0.0.1
  gpsd_port: 2947
```

To configure `gpsd` for a soldered UART GPS on `/dev/serial0`, edit `/etc/default/gpsd`:

```bash
sudo nano /etc/default/gpsd
```

Use:

```bash
DEVICES="/dev/serial0"
GPSD_OPTIONS="-n"
USBAUTO="false"
```

Then restart and test:

```bash
sudo systemctl stop gpsd.socket gpsd
sudo killall gpsd 2>/dev/null
sudo systemctl enable gpsd.socket
sudo systemctl restart gpsd.socket
cgps
```

When a human is detected, the dashboard pins the current drone/camera GPS position and keeps that "last seen" pin for 60 seconds after the last matching detection. Estimating the detected person's actual ground coordinate requires drone altitude, camera angle, field of view calibration, and a ground-plane projection.

## Logging

When enabled, track events are appended to JSONL:

```json
{"timestamp":1710000000.0,"frame_number":42,"track_id":1,"bbox":[100,120,240,420],"confidence":0.91,"label":"human"}
```

Only live, matched tracks are logged each frame. Lost tracks are retained internally for ID persistence but are not written as fresh detections.

## Troubleshooting Camera Issues

- Run `rpicam-hello --list-cameras` to confirm the IMX708 is detected.
- Set `camera.source: rpicam` in your local Pi config if Picamera2 is unavailable but `rpicam-hello` works.
- Confirm the ribbon cable orientation and seating.
- Use a recent Raspberry Pi OS 64-bit image.
- Ensure no other process is using the camera.
- Try a lower resolution/FPS in your local Pi config.
- If the camera view feels too tight, try `1280x720` for 16:9 or `1024x768` for a taller 4:3 view before moving the camera.
- If Picamera2 import fails, install `python3-picamera2` with apt rather than pip.
- For development without camera hardware, use `--backend mock`; the app will fall back to synthetic frames.

## Safety And Legal Disclaimer

Operate drones safely, legally, and with respect for privacy. Follow local aviation rules, obtain required permissions, avoid flying over people without authorization, and do not use this system for surveillance where people have a reasonable expectation of privacy. This prototype is not a certified flight-safety or collision-avoidance system.

## Tests

```bash
pytest
```

The included tests cover configuration loading and centroid tracker ID behavior.
GitHub Actions runs the test suite on Python 3.11 and 3.12 for pushes to `main` and pull requests.

## Future Upgrade Ideas

- MAVLink telemetry integration
- GPS tagging of detections
- Object re-identification across occlusions
- Drone ground-station UI
- Search-and-rescue mode with heatmap/history trails
- Real Hailo post-processing adapters for specific YOLO/SSD models
- On-device recording rotation and mission metadata
