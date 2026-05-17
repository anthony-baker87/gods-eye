from src.config import GpsConfig
from src.gps import DisabledGpsSource, StaticGpsSource, create_gps_source


def test_disabled_gps_source_returns_no_location() -> None:
    source = create_gps_source(GpsConfig(enabled=False))

    assert isinstance(source, DisabledGpsSource)
    assert source.read() is None


def test_static_gps_source_returns_configured_location() -> None:
    source = create_gps_source(
        GpsConfig(
            enabled=True,
            provider="static",
            latitude=34.05,
            longitude=-118.25,
            altitude_m=120.0,
        )
    )

    assert isinstance(source, StaticGpsSource)
    location = source.read()
    assert location.latitude == 34.05
    assert location.longitude == -118.25
    assert location.altitude_m == 120.0
    assert location.source == "static"
