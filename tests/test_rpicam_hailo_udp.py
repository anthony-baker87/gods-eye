import struct

from src.detector.hailo_detector import _parse_object_detect_udp_packet, _scale_udp_detection


def test_parse_object_detect_udp_person_packet() -> None:
    packet = bytearray(280)
    struct.pack_into("<Iiiii", packet, 0, 0xABCDEF01, 10, 20, 30, 40)
    packet[20] = len("person")
    packet[21:27] = b"person"
    struct.pack_into("<f", packet, 276, 0.72)

    detection = _parse_object_detect_udp_packet(bytes(packet))

    assert detection is not None
    assert detection.bbox == (10, 20, 40, 60)
    assert detection.class_id == 0
    assert detection.label == "human"
    assert round(detection.confidence, 2) == 0.72


def test_parse_object_detect_udp_non_person_packet() -> None:
    packet = bytearray(280)
    struct.pack_into("<Iiiii", packet, 0, 0xABCDEF01, 1, 2, 3, 4)
    packet[20] = len("airplane")
    packet[21:29] = b"airplane"
    struct.pack_into("<f", packet, 276, 0.67)

    detection = _parse_object_detect_udp_packet(bytes(packet))

    assert detection is not None
    assert detection.class_id == -1
    assert detection.label == "airplane"


def test_scale_udp_detection_to_frame_size() -> None:
    packet = bytearray(280)
    struct.pack_into("<Iiiii", packet, 0, 0xABCDEF01, 160, 160, 160, 160)
    packet[21:27] = b"person"
    struct.pack_into("<f", packet, 276, 0.9)
    detection = _parse_object_detect_udp_packet(bytes(packet))

    assert detection is not None
    scaled = _scale_udp_detection(detection, source_width=640, source_height=640, frame_width=1280, frame_height=720)

    assert scaled.bbox == (320, 180, 640, 360)
