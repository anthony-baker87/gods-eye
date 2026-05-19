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
        self.timeout_seconds = 0.05
        self._socket: socket.socket | None = None
        self._buffer = ""
        self._latest: GpsLocation | None = None

    def read(self) -> GpsLocation | None:
        self._ensure_connected()
        if self._socket is None:
            return self._latest

        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            try:
                chunk = self._socket.recv(4096).decode("utf-8", errors="ignore")
            except TimeoutError:
                break
            except OSError as exc:
                LOGGER.debug("gpsd read failed: %s", exc)
                self._close()
                break
            if not chunk:
                self._close()
                break
            self._buffer += chunk
            lines = self._buffer.splitlines(keepends=True)
            self._buffer = ""
            for line in lines:
                if line.endswith("\n") or line.endswith("\r"):
                    location = self._parse_line(line.strip())
                    if location is not None:
                        self._latest = location
                else:
                    self._buffer = line
        return self._latest

    def _ensure_connected(self) -> None:
        if self._socket is not None:
            return
        try:
            self._socket = socket.create_connection((self.host, self.port), timeout=self.timeout_seconds)
            self._socket.settimeout(self.timeout_seconds)
            self._socket.sendall(b'?WATCH={"enable":true,"json":true};\n')
        except OSError as exc:
            LOGGER.debug("gpsd read failed: %s", exc)
            self._close()

    def _close(self) -> None:
        if self._socket is None:
            return
        try:
            self._socket.close()
        finally:
            self._socket = None
            self._buffer = ""

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
