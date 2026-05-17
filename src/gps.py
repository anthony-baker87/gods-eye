from __future__ import annotations

import json
import logging
import socket
import time
from dataclasses import dataclass

from src.config import GpsConfig

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class GpsLocation:
    latitude: float
    longitude: float
    altitude_m: float | None = None
    timestamp: float = 0.0
    source: str = "unknown"

    def as_dict(self) -> dict[str, float | str | None]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude_m": self.altitude_m,
            "timestamp": self.timestamp,
            "source": self.source,
        }


class GpsSource:
    def read(self) -> GpsLocation | None:
        raise NotImplementedError


class DisabledGpsSource(GpsSource):
    def read(self) -> GpsLocation | None:
        return None


class StaticGpsSource(GpsSource):
    def __init__(self, config: GpsConfig) -> None:
        if config.latitude is None or config.longitude is None:
            raise ValueError("Static GPS requires latitude and longitude.")
        self.location = GpsLocation(
            latitude=config.latitude,
            longitude=config.longitude,
            altitude_m=config.altitude_m,
            timestamp=time.time(),
            source="static",
        )

    def read(self) -> GpsLocation:
        self.location.timestamp = time.time()
        return self.location


class GpsdGpsSource(GpsSource):
    def __init__(self, config: GpsConfig) -> None:
        self.host = config.gpsd_host
        self.port = config.gpsd_port
        self.timeout_seconds = 0.25

    def read(self) -> GpsLocation | None:
        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout_seconds) as sock:
                sock.settimeout(self.timeout_seconds)
                sock.sendall(b'?WATCH={"enable":true,"json":true};\n')
                deadline = time.monotonic() + self.timeout_seconds
                buffer = ""
                while time.monotonic() < deadline:
                    chunk = sock.recv(4096).decode("utf-8", errors="ignore")
                    if not chunk:
                        break
                    buffer += chunk
                    for line in buffer.splitlines():
                        location = self._parse_line(line)
                        if location is not None:
                            return location
        except OSError as exc:
            LOGGER.debug("gpsd read failed: %s", exc)
        return None

    def _parse_line(self, line: str) -> GpsLocation | None:
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            return None
        if message.get("class") != "TPV":
            return None
        if "lat" not in message or "lon" not in message:
            return None
        mode = int(message.get("mode", 0))
        if mode < 2:
            return None
        return GpsLocation(
            latitude=float(message["lat"]),
            longitude=float(message["lon"]),
            altitude_m=float(message["alt"]) if "alt" in message else None,
            timestamp=time.time(),
            source="gpsd",
        )


def create_gps_source(config: GpsConfig) -> GpsSource:
    if not config.enabled:
        return DisabledGpsSource()
    if config.provider == "static":
        return StaticGpsSource(config)
    if config.provider == "gpsd":
        return GpsdGpsSource(config)
    raise ValueError(f"Unknown GPS provider: {config.provider}")
