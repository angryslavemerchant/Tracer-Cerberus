# HailoRT Python API (pyHailoRT) Reference

## Overview

HailoRT uses the `hailo_platform` package (pyHailoRT) to run inference on Hailo AI accelerators.
Models are compiled to `.hef` files using the Hailo Dataflow Compiler before runtime use.

The inference flow is: **Load model → Configure device → Set up vstreams → Run inference**

---

## Minimal Inference Example

```python
import numpy as np
import hailo_platform as hpf

# 1. Load compiled model
hef = hpf.HEF("my_model.hef")

# 2. Open device and configure
with hpf.VDevice() as target:
    configure_params = hpf.ConfigureParams.create_from_hef(
        hef, interface=hpf.HailoStreamInterface.PCIe
    )
    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()

    # 3. Inspect input/output shapes
    input_info  = hef.get_input_vstream_infos()[0]
    output_info = hef.get_output_vstream_infos()[0]
    print(f"Input:  {input_info.name}  shape={input_info.shape}")
    print(f"Output: {output_info.name} shape={output_info.shape}")

    # 4. Create vstream params (use FLOAT32 for unquantized models)
    input_params = hpf.InputVStreamParams.make_from_network_group(
        network_group, quantized=False, format_type=hpf.FormatType.FLOAT32
    )
    output_params = hpf.OutputVStreamParams.make_from_network_group(
        network_group, quantized=False, format_type=hpf.FormatType.FLOAT32
    )

    # 5. Run inference
    with network_group.activate(network_group_params):
        with hpf.InferVStreams(network_group, input_params, output_params) as pipeline:
            data = np.random.rand(*input_info.shape).astype(np.float32)
            input_data = {input_info.name: np.expand_dims(data, axis=0)}  # add batch dim
            results = pipeline.infer(input_data)
            output = results[output_info.name]
```

---

## ONNX Runtime Analogy

| ONNX Runtime | HailoRT |
|---|---|
| `ort.InferenceSession("model.onnx")` | `hpf.HEF("model.hef")` + `target.configure(...)` |
| `session.get_inputs()[0]` | `hef.get_input_vstream_infos()[0]` |
| `session.get_outputs()[0]` | `hef.get_output_vstream_infos()[0]` |
| `session.run(None, {"input": data})` | `pipeline.infer({"input_name": data})` |
| Returns list | Returns dict keyed by output name |

---

## Key Concepts

### Model Format
- `.hef` — Hailo Executable Format; compiled output from the Hailo Dataflow Compiler
- Equivalent to `.trt` (TensorRT) or `.tflite` (TFLite)

### VDevice
- `hpf.VDevice()` — context manager that opens the Hailo device
- For multi-device setups, configure scheduling via `HailoSchedulingAlgorithm` in VDeviceParams

### Quantization / Format Type
- `quantized=False, format_type=hpf.FormatType.FLOAT32` — feed float32; device handles dequantization internally
- `quantized=True` — feed raw uint8 quantized inputs (faster, skips conversion)

### VStreams (Virtual Streams)
- `InputVStreamParams` / `OutputVStreamParams` — configure how data flows to/from the device
- `InferVStreams` — synchronous inference pipeline; simplest path, good for getting started
- For high-throughput use cases, switch to the async queue-based pipeline (`HailoAsyncInference`)

### Input/Output Dict Keys
- Inputs and outputs are keyed by **name** (`vstream_info.name`)
- Always add a batch dimension: `np.expand_dims(data, axis=0)`

---

## Multiple Inputs/Outputs

```python
input_infos  = hef.get_input_vstream_infos()
output_infos = hef.get_output_vstream_infos()

# Build input dict for all inputs
input_data = {
    info.name: np.expand_dims(np.random.rand(*info.shape).astype(np.float32), axis=0)
    for info in input_infos
}

results = pipeline.infer(input_data)

# Access each output by name
for info in output_infos:
    print(info.name, results[info.name].shape)
```

---

## Notes & Gotchas

- Official full docs require registration at hailo.ai/developer-zone
- Most official examples are heavy/abstracted around pre-trained model zoo workflows — the pattern above is the bare-bones equivalent
- Tested on HailoRT v4.19.0 with Hailo-8 / Hailo-8L (e.g. Raspberry Pi 5 AI HAT)
- There is no pip-installable `hailo_platform` from PyPI — it ships with the HailoRT runtime `.deb` install
- Python version must match the available wheel bundled with your HailoRT version
- `quantized=False, FLOAT32` expects pixel values in **[0, 255]** range, not [0, 1] — the model's calibration params are anchored to uint8 range
- Input format is **NHWC** `(1, H, W, C)` on Hailo, not NCHW like PyTorch/ONNX

---

## YOLOv8s (Hailo Model Zoo) — Confirmed Behaviour

- HEF: `Models/yolov8s.hef`
- Input: `yolov8s/input_layer1`, shape `(640, 640, 3)` NHWC
- Output: `yolov8s/yolov8_nms_postprocess`, NMS baked in
- Output structure: `results[name]` returns a **list of length 1** (batch), where `result[0]` is a **list of 80 arrays** (one per COCO class), each shaped `(N, 5)` — `[y1, x1, y2, x2, score]` with normalized coords `[0, 1]`
- Empty arrays `(0, 5)` mean no detections for that class — normal

---

## picamera2 + OpenCV Colour Order (Raspberry Pi 5, imx708_noir)

- `picamera2` with `"RGB888"` format actually stores pixels in **BGR order** in memory on this system (libcamera/PiSP quirk)
- Display with `cv2.imshow` directly — no conversion needed
- Convert to RGB before model inference: `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`
- Camera in use: **Camera Module 3 Noir** (`imx708_noir`)
- Camera is mounted upside-down — use `Transform(hflip=True, vflip=True)` in picamera2 config (ISP handles it for free, ~3ms cheaper than `cv2.ROTATE_180`)

---

## Pipeline Performance (Raspberry Pi 5 + Hailo-8L, YOLOv8s, 1280x720)

- Sustained **~55-65 FPS** headless (no display)
- `cv2.imshow` + `waitKey` locks to ~30fps via vsync — run display in a separate thread or skip for max throughput
- Per-step breakdown: capture ~2ms, cvt ~1ms, preprocess ~2ms, infer ~11ms, draw ~1.5ms
- Input as `uint8` (`quantized=True, UINT8`) vs float32 shows marginal difference — float32 is fine
- imx708 supports 120fps at 1280x720 — set via `controls={"FrameRate": 120}` in picamera2 config

---

## References

- GitHub: https://github.com/hailo-ai/hailort
- Community examples: https://github.com/hailo-ai/Hailo-Application-Code-Examples/tree/main/runtime/python
- Annotated example: https://github.com/TeigLevingston/Hailo_Engine_Example
- Community forum: https://community.hailo.ai
