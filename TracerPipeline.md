# Tracer-Cerberus Pipeline

## Overview

Two-stage detect-then-track pipeline running entirely on Raspberry Pi 5 + Hailo-8L AI HAT.

- **Stage 1 (Acquisition):** YOLOv8s on Hailo detects the target class and cuts a template crop
- **Stage 2 (Tracking):** CerberusCore (LightTrack Siamese network) on Hailo tracks the target frame-to-frame using heatmap peak + PSR gating
- Both models run on the same Hailo device simultaneously via `ROUND_ROBIN` scheduling

---

## Hardware

- Raspberry Pi 5
- Hailo-8L AI HAT (13 TOPS)
- Camera Module 3 Noir (`imx708_noir`), mounted upside-down — flipped via ISP `Transform(hflip=True, vflip=True)`

---

## Models

| Model | File | Input | Output |
|---|---|---|---|
| YOLOv8s | `Models/yolov8s.hef` | `(1, 640, 640, 3)` uint8 NHWC | list of 80 arrays `(N, 5)` — `[y1, x1, y2, x2, score]` normalized |
| CerberusCore | `Models/CerberusCoreS.hef` | template `(1, 128, 128, 3)` + search `(1, 256, 256, 3)` float32 NHWC | `(1, 16, 16, 1)` raw logits |

**Preprocessing notes:**
- YOLOv8s: `quantized=True, UINT8` — feed raw uint8, Hailo quantizes internally
- CerberusCore: `quantized=False, FLOAT32` — feed `[0, 255]` float32; **ImageNet normalization is baked into the HEF** as a first-layer op (do not normalize externally)
- Both models use NHWC layout, not NCHW

---

## State Machine

```
SEARCHING
    │  YOLO detects target class
    │  → cut template, run Cerberus, check PSR
    ▼
TRACKING  ◄──────────────────────────────┐
    │  Cerberus runs every frame          │
    │  PSR >= PSR_THRESH → update bbox    │
    │  PSR <  PSR_THRESH ─────────────────┘
    ▼
REACQUIRING
    │  same as SEARCHING
    └──► TRACKING (on next successful lock)
```

Press `r` in the display window to manually reset to SEARCHING.

---

## Cerberus Output Decoding

CerberusCore outputs a `(16, 16)` score map (raw logits). No regression head — position only.

```
raw (1,16,16,1)
  → squeeze → (16,16)
  → sigmoid → score map [0,1]
  → compute_psr() → PSR + peak (row, col)
  → _GRID_X/_GRID_Y[peak_r, peak_c] → (cx, cy) in 256px search coords
  → heatmap_center_to_frame() → bbox in frame coords
```

Box size is fixed to `template_size_px` (set at acquisition, carried forward — no regression-based resize).

---

## PSR (Peak-to-Sidelobe Ratio)

Measures sharpness of the response peak relative to the background noise. Robust to cases where raw sigmoid confidence looks high but the target is actually absent (flat response map).

- `< 7` — target likely gone, reacquire
- `7–10` — marginal / partial occlusion
- `> 10` — solid lock
- `PSR_THRESH = 18` (current setting)

---

## Search Window

Fixed at acquisition — set once as `max(template_size_px * SEARCH_SCALE_INIT, MIN_SEARCH_PX)` and held constant for the life of the track. No adaptive resizing.

---

## Key Config

```python
CLASSES_TO_DETECT = [14]    # COCO class index — 14 = bird
YOLO_CONF         = 0.5
PSR_THRESH        = 18
CROP_SIZE         = 640     # working area cropped from centre of 1280x720
SEARCH_SCALE_INIT = 2.0
MIN_SEARCH_PX     = 64
```

---

## Threading

Display runs on a separate thread (`display_thread`) so `cv2.imshow` / `cv2.waitKey` vsync does not block the inference loop. Shared frame is protected by a `threading.Lock`. Without this, vsync locks the loop to ~30fps; with threading the inference loop runs at full throughput (~40-60fps depending on state).

---

## Multi-Model on Hailo

Both models are configured on the same `VDevice` using `HailoSchedulingAlgorithm.ROUND_ROBIN`. Both `InferVStreams` contexts are kept open for the life of the loop — `pipeline.infer()` is called on the appropriate one based on state. No manual `activate()` / `deactivate()` needed with ROUND_ROBIN.

```python
vdevice_params = hpf.VDevice.create_params()
vdevice_params.scheduling_algorithm = hpf.HailoSchedulingAlgorithm.ROUND_ROBIN
with hpf.VDevice(vdevice_params) as target:
    ng_yolo     = target.configure(hef_yolo, ...)[0]
    ng_cerberus = target.configure(hef_cerberus, ...)[0]
    # open both InferVStreams, call infer on whichever is needed per frame
```

---

## Error Signal

`error_x` / `error_y` — pixel offset of tracked bbox centre from `frame_center`. Ready to feed directly into a gimbal PID controller.

```python
error_x = tracked_cx - frame_center[0]
error_y = tracked_cy - frame_center[1]
```
