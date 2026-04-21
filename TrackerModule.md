# Tracker Module

## Overview

The tracker is split into three files:

| File | Role |
|---|---|
| `tracker.py` | `TrackerHeadless` — all tracking logic, no display |
| `tracker_display.py` | `TrackerDisplay(TrackerHeadless)` — adds threaded display |
| `main.py` | Camera setup, main loop, wires tracker to downstream pipeline |

Swapping between headless and display is a one-line change in `main.py`.

---

## Interface

Both classes share the same interface:

```python
tracker = TrackerDisplay()   # or TrackerHeadless()

error_x, error_y, state = tracker.update(frame)

tracker.reset()   # force back to SEARCHING
tracker.stop()    # clean shutdown
tracker.should_stop()  # returns True if 'q' was pressed (display only)
```

`update(frame)` accepts a raw BGR frame from picamera2 (1280×720) and returns:
- `error_x` — horizontal pixel offset of target from frame center (negative = left)
- `error_y` — vertical pixel offset of target from frame center (negative = up)
- `state` — current `TrackState` string (`SEARCHING`, `TRACKING`, `REACQUIRING`)

When not tracking, `error_x` and `error_y` are both `0`.

---

## main.py Structure

```python
picam2 = Picamera2()
# ... configure 1280x720 RGB888, 120fps, ISP flip, continuous AF ...
picam2.start()

tracker = TrackerDisplay()
# swap to: from tracker import TrackerHeadless as Tracker

try:
    while not tracker.should_stop():
        frame = picam2.capture_array()
        error_x, error_y, state = tracker.update(frame)

        # TODO: coords_to_rads(error_x, error_y)
        # TODO: send_to_mcu(rads)
finally:
    tracker.stop()
    picam2.stop()
```

---

## Tracker Internals

### Models

Both models run on the Hailo-8L via `ROUND_ROBIN` scheduling on a single `VDevice`. Both `InferVStreams` contexts stay open for the life of the tracker.

| Model | Purpose | Input | Output |
|---|---|---|---|
| YOLOv8s | Target acquisition | 1280×720 → resized 640×640 uint8 | Per-class detection arrays |
| CerberusCore | Frame-to-frame tracking | template 128×128 + search 256×256 float32 | 16×16 score map |

### State Machine

```
SEARCHING / REACQUIRING
    YOLO runs every frame looking for CLASSES_TO_DETECT
    On detection → cut template crop → run Cerberus → check PSR
    PSR >= PSR_THRESH → TRACKING

TRACKING
    Cerberus runs every frame on search crop centred on last bbox
    PSR >= PSR_THRESH → update bbox
    PSR <  PSR_THRESH → REACQUIRING
```

### Cerberus Decoding

1. Raw output `(1, 16, 16, 1)` → sigmoid → `(16, 16)` score map
2. PSR computed on the 16×16 map for presence gating
3. Map upsampled to `256×256` via bicubic interpolation
4. Argmax on upsampled map → pixel-resolution peak `(px, py)` in search coords
5. Scaled back to frame coordinates via `_heatmap_center_to_frame()`

Box size is fixed to `template_size_px` (set at acquisition, not updated during tracking).

### Key Config

```python
CLASSES_TO_DETECT = [14]    # COCO index — 14 = bird
YOLO_CONF         = 0.5
PSR_THRESH        = 18
TEMPLATE_SIZE     = 128     # px
SEARCH_SIZE       = 256     # px
SEARCH_SCALE_INIT = 2.0     # search window = template_size * this
MIN_SEARCH_PX     = 64
```

---

## TrackerDisplay

Inherits `TrackerHeadless`. Adds:

- A background display thread that calls `cv2.imshow` independently — inference loop is never blocked by display rendering
- `update()` drops a raw frame + state snapshot into a shared buffer; the display thread draws on its own cadence
- `q` in the display window sets the stop event; `r` resets to SEARCHING
- Overlays: heatmap, search window box, tracked bbox + crosshair, PSR bar, stats

---

## Error Signal

```python
error_x = tracked_cx - frame_center_x   # positive = target right of centre
error_y = tracked_cy - frame_center_y   # positive = target below centre
```

`frame_center` is computed from the actual frame size on each call, so it adapts automatically if resolution changes. Both errors are `0` when not tracking.

Next step: `coords_to_rads(error_x, error_y)` → motor driver.
