# God's Eye

Python prototype for onboard person detection and tracking on a Raspberry Pi 5 16GB with the Raspberry Pi AI Kit / Hailo-8L accelerator and an IMX708 wide camera. The app captures frames with Picamera2 or `rpicam-vid`, detects people, assigns persistent track IDs, draws overlays, logs track events, tags confirmed detections with GPS, and can serve a local MJPEG dashboard.

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
    snapshots.py
    suppression.py
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

Install the Raspberry Pi AI Kit / Hailo software using Raspberry Pi and Hailo's current official instructions for your OS image. This app uses the Raspberry Pi `rpicam-apps` Hailo post-processing path for the practical Pi backend because it is the same path used by `rpicam-hello --post-process-file`.

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

The CPU backend uses OpenCV face detection by default. It is included as a development fallback, not as the target real-time path:

```bash
python -m src.main --config config.pi.yaml --backend cpu --debug
```

Expect lower accuracy and performance than a modern Hailo model. It works best with good lighting and a visible face. The older OpenCV HOG full-body detector is disabled by default because it can mistake vertical hardware, furniture, or bright edges for people. To opt into it for outdoor experiments, set `detection.cpu_full_body: true`.

The CPU backend downsizes frames internally for inference, so you can use a larger camera stream such as 1280x720 for the dashboard without making CPU detection scale linearly with video size.

## Run With Hailo Backend

First verify the Raspberry Pi Hailo post-process pipeline sees people:

```bash
rpicam-hello -t 0 \
  --post-process-file /usr/share/rpi-camera-assets/hailo_yolov8_inference.json \
  --lores-width 640 \
  --lores-height 640
```

Stop that preview with `Ctrl+C`, then create the app's UDP-enabled post-process file:

```bash
python tools/make_rpicam_hailo_udp_config.py \
  --source /usr/share/rpi-camera-assets/hailo_yolov8_inference.json \
  --output hailo_yolov8_udp.json
```

Set the Pi config to use `rpicam_hailo`:

```yaml
camera:
  width: 1280
  height: 720
  fps: 30
  source: rpicam

detection:
  backend: rpicam_hailo
  confidence_threshold: 0.45
  hailo:
    post_process_file: hailo_yolov8_udp.json
    udp_host: 127.0.0.1
    udp_port: 12347
    lores_width: 640
    lores_height: 640
```

Then run:

```bash
python -m src.main --config config.pi.yaml --backend rpicam_hailo
```

This starts `rpicam-vid` for dashboard video and asks `rpicam-apps` to run Hailo YOLO on the low-resolution stream. The `object_detect_udp` stage sends boxes to the app on localhost, and the app only keeps `person` detections before tracking, GPS pinning, logging, and snapshots.

The older direct PyHailoRT adapter is still available as `--backend hailo`, but the current recommended Pi backend is `rpicam_hailo`.

If PyHailoRT rejects the input buffer while bringing up a new model/runtime combination, run:

```bash
python tools/hailo_input_probe.py --model /usr/share/hailo-models/yolov8s_h8l.hef
```

The probe tries common PyHailoRT input formats and prints the first one accepted by the installed runtime.

## Configuration

`config.yaml` controls:

- Camera width, height, and FPS
- Camera source: `auto`, `picamera2`, `rpicam`, or `synthetic`
- Detector backend: `rpicam_hailo`, `hailo`, `cpu`, or `mock`
- Confidence threshold
- Optional CPU full-body HOG detector toggle: `detection.cpu_full_body`
- Optional normalized suppression zones for fixed false-positive regions
- Tracker maximum lost frames and matching distance
- Track confirmation frames, minimum smoothed confidence, and confidence smoothing
- Dashboard host, port, and JPEG quality
- Optional GPS provider for detection map pins: disabled, static test coordinates, or `gpsd`
- JSONL logging enabled/path
- Optional annotated video recording path
- Snapshot output for confirmed detection events

CLI overrides:

```bash
python -m src.main --config config.yaml --backend mock --record-output --debug
```

## Dashboard Endpoints

- `/` shows MJPEG video with overlays and live status JSON.
- Click the camera video to drop a manual map pin at the current GPS location. The pin stores the clicked image pixel, timestamp, and snapshot.
- The dashboard map places a "last seen" pin when a human is detected and GPS is enabled.
- Click a detection or manual pin to show its timestamp, source, confidence when available, and snapshot when available.
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
  "manual_pins": [],
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

When a human track is detected for several consecutive frames, the dashboard pins the current drone/camera GPS position and keeps that "last seen" pin for 60 seconds after the last matching detection. Estimating the detected person's actual ground coordinate requires drone altitude, camera angle, field of view calibration, and a ground-plane projection.

Manual camera-click pins use the same current drone/camera GPS position. They are useful for marking what the operator saw in the camera view, but they are not yet projected to the clicked object's ground coordinate.

## Detection Quality Controls

Tracks are only reported after repeated evidence. The defaults require three matched frames and a smoothed confidence of at least `0.7`:

```yaml
tracking:
  max_lost_frames: 12
  max_distance: 90
  confirmation_frames: 3
  min_confirmed_confidence: 0.7
  confidence_smoothing: 0.35
```

Use suppression zones to ignore fixed camera regions that contain the drone body, propellers, landing gear, or other repeat false positives. Coordinates are normalized from `0.0` to `1.0` across the frame:

```yaml
detection:
  suppression_zones:
    - name: drone_body_bottom_left
      x1: 0.0
      y1: 0.65
      x2: 0.25
      y2: 1.0
```

Confirmed detections save a full-frame snapshot, crop, and JSON metadata once per track:

```yaml
snapshots:
  enabled: true
  path: output/snapshots
  jpeg_quality: 90
```

## Logging

When enabled, track events are appended to JSONL:

```json
{"timestamp":1710000000.0,"frame_number":42,"track_id":1,"bbox":[100,120,240,420],"confidence":0.91,"smoothed_confidence":0.88,"confirmed":true,"label":"human"}
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
