from __future__ import annotations

import argparse
from typing import Any

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe PyHailoRT input formats.")
    parser.add_argument("--model", default="/usr/share/hailo-models/yolov8s_h8l.hef")
    args = parser.parse_args()

    import hailo_platform as hailo

    hef = hailo.HEF(args.model)
    target = hailo.VDevice()
    configure_params = hailo.ConfigureParams.create_from_hef(hef, interface=hailo.HailoStreamInterface.PCIe)
    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()
    input_info = hef.get_input_vstream_infos()[0]
    output_infos = hef.get_output_vstream_infos()
    shape = tuple(int(item) for item in input_info.shape)
    height, width = int(shape[0]), int(shape[1])

    print(f"input name: {input_info.name}")
    print(f"input shape: {shape}")
    print(f"outputs: {[info.name for info in output_infos]}")

    frames = [
        ("dict_batched_uint8", lambda: {input_info.name: np.zeros((1, height, width, 3), dtype=np.uint8)}),
        ("dict_list_uint8", lambda: {input_info.name: [np.zeros((height, width, 3), dtype=np.uint8)]}),
        ("dict_single_uint8", lambda: {input_info.name: np.zeros((height, width, 3), dtype=np.uint8)}),
        ("array_batched_uint8", lambda: np.zeros((1, height, width, 3), dtype=np.uint8)),
        ("array_single_uint8", lambda: np.zeros((height, width, 3), dtype=np.uint8)),
        ("dict_batched_float32", lambda: {input_info.name: np.zeros((1, height, width, 3), dtype=np.float32)}),
    ]

    param_sets: list[tuple[str, Any, Any]] = [
        (
            "quantized_uint8",
            hailo.InputVStreamParams.make_from_network_group(
                network_group,
                quantized=True,
                format_type=hailo.FormatType.UINT8,
            ),
            hailo.OutputVStreamParams.make_from_network_group(
                network_group,
                quantized=False,
                format_type=hailo.FormatType.FLOAT32,
            ),
        ),
        (
            "float32",
            hailo.InputVStreamParams.make_from_network_group(
                network_group,
                quantized=False,
                format_type=hailo.FormatType.FLOAT32,
            ),
            hailo.OutputVStreamParams.make_from_network_group(
                network_group,
                quantized=False,
                format_type=hailo.FormatType.FLOAT32,
            ),
        ),
    ]

    for params_name, input_params, output_params in param_sets:
        print(f"\n== {params_name} ==")
        with network_group.activate(network_group_params):
            with hailo.InferVStreams(network_group, input_params, output_params) as infer_pipeline:
                for case_name, make_input in frames:
                    payload = make_input()
                    try:
                        result = infer_pipeline.infer(payload)
                    except Exception as exc:
                        print(f"FAIL {case_name}: {type(exc).__name__}: {exc}")
                        continue
                    print(f"OK   {case_name}: {[name + ':' + str(np.asarray(value).shape) for name, value in result.items()]}")
                    return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
