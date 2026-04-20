import cv2
from ultralytics import YOLO
import numpy as np
import onnxruntime as ort
import time
from threading import Thread


# ── Threaded Camera ───────────────────────────────────────────────────────────

class ThreadedCamera:
    def __init__(self, src=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera: {actual_w}x{actual_h}")

        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.ret:
                self.stop()
            else:
                self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()


# ── LightTrack constants ──────────────────────────────────────────────────────

MEAN          = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD           = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
TEMPLATE_SIZE = 128
SEARCH_SIZE   = 256
STRIDE        = 16
SCORE_SIZE    = SEARCH_SIZE // STRIDE   # 16


# ── LightTrack helpers ────────────────────────────────────────────────────────

def preprocess(img_bgr: np.ndarray, out_size: int) -> np.ndarray:
    """BGR uint8 HxWx3 → float32 1x3xHxW (ImageNet-normalised)."""
    img_rgb     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (out_size, out_size))
    x = img_resized.astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)
    x = (x - MEAN) / STD
    return x[np.newaxis]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


def make_grid(score_size: int, stride: int, search_size: int):
    half    = score_size // 2
    xs      = np.arange(0, score_size) - np.floor(float(half))
    ys      = np.arange(0, score_size) - np.floor(float(half))
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_x  = grid_x * stride + search_size // 2
    grid_y  = grid_y * stride + search_size // 2
    return grid_x, grid_y


def decode_boxes(reg: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    x1 = grid_x - reg[0]
    y1 = grid_y - reg[1]
    x2 = grid_x + reg[2]
    y2 = grid_y + reg[3]
    return np.stack([x1, y1, x2, y2], axis=-1)


_GRID_X, _GRID_Y = make_grid(SCORE_SIZE, STRIDE, SEARCH_SIZE)


def compute_psr(score_map: np.ndarray, exclude_radius: int = 2) -> tuple:
    """
    Peak-to-Sidelobe Ratio.

    Measures how sharply the response peak stands out from the rest of the
    map.  When the target is present, cross-correlation produces a tight
    spike → high PSR.  When the target is absent, the map is flat or noisy
    → low PSR, even if the raw sigmoid peak value looks similar.

    This is why raw sigmoid peak fails as a presence detector but PSR works.

    Parameters
    ----------
    score_map      : (S, S) float32 sigmoid scores from LightTrack
    exclude_radius : cells around the peak to exclude from sidelobe stats.
                     On a 16×16 map, radius=2 gives a 5×5 exclusion zone.

    Returns
    -------
    psr     : float  — rough guide:  <7 lost  |  7-10 marginal  |  >10 solid
    peak_rc : (row, col) of the peak cell
    """
    peak_idx       = int(np.argmax(score_map))
    peak_r, peak_c = np.unravel_index(peak_idx, score_map.shape)

    h, w = score_map.shape
    mask = np.ones_like(score_map, dtype=bool)
    r0   = max(0, peak_r - exclude_radius)
    r1   = min(h, peak_r + exclude_radius + 1)
    c0   = max(0, peak_c - exclude_radius)
    c1   = min(w, peak_c + exclude_radius + 1)
    mask[r0:r1, c0:c1] = False

    sidelobe = score_map[mask]
    psr      = (score_map[peak_r, peak_c] - sidelobe.mean()) / (sidelobe.std() + 1e-6)

    return float(psr), (peak_r, peak_c)


def run_lighttrack(sess, template_tensor: np.ndarray, search_bgr: np.ndarray):
    """
    Returns
    -------
    best_box  : (x1, y1, x2, y2) in SEARCH_SIZE (256-px) coordinates
    psr       : float  — use this to gate tracking / reacquisition
    raw_conf  : float  — raw sigmoid peak value (display only)
    score_map : (16, 16) sigmoid score map
    """
    search_tensor    = preprocess(search_bgr, SEARCH_SIZE)
    cls_raw, reg_raw = sess.run(None, {"template": template_tensor,
                                       "search":   search_tensor})
    cls_sig          = sigmoid(cls_raw[0, 0]).astype(np.float32)   # (16, 16)
    reg              = reg_raw[0]                                   # (4, 16, 16)
    boxes            = decode_boxes(reg, _GRID_X, _GRID_Y)         # (16, 16, 4)

    psr, (peak_r, peak_c) = compute_psr(cls_sig)

    # Derive the bbox from the same peak cell used for PSR so they're consistent
    flat_boxes = boxes.reshape(-1, 4)
    best_idx   = peak_r * cls_sig.shape[1] + peak_c
    best_box   = flat_boxes[best_idx]
    raw_conf   = float(cls_sig[peak_r, peak_c])

    return best_box, psr, raw_conf, cls_sig


# ── Crop helpers ──────────────────────────────────────────────────────────────

def extract_square_crop(frame: np.ndarray, cx: int, cy: int, size: int):
    """
    Crop a (size × size) region centred at (cx, cy).
    Pads with zeros where the region extends beyond frame boundaries.
    Returns (crop, (ox, oy)) where ox/oy is the top-left in frame coords.
    """
    h, w   = frame.shape[:2]
    half   = size // 2
    ox, oy = cx - half, cy - half

    x1c    = max(0, ox);           y1c = max(0, oy)
    x2c    = min(w, ox + size);    y2c = min(h, oy + size)

    crop   = np.zeros((size, size, 3), dtype=np.uint8)
    dst_x  = x1c - ox
    dst_y  = y1c - oy
    crop[dst_y : dst_y + (y2c - y1c),
         dst_x : dst_x + (x2c - x1c)] = frame[y1c:y2c, x1c:x2c]

    return crop, (ox, oy)


def search_box_to_frame(best_box_search, origin, search_size_px):
    """Map a box from SEARCH_SIZE coords back to frame coords → (x, y, w, h)."""
    scale  = search_size_px / SEARCH_SIZE
    ox, oy = origin
    x1 = ox + best_box_search[0] * scale
    y1 = oy + best_box_search[1] * scale
    x2 = ox + best_box_search[2] * scale
    y2 = oy + best_box_search[3] * scale
    return int(x1), int(y1), int(x2 - x1), int(y2 - y1)


# ── YOLO helper ───────────────────────────────────────────────────────────────

def best_detection(results, frame_center, classes):
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    best_box, best_dist = None, float('inf')
    for box in boxes:
        if int(box.cls[0]) not in classes:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        dist = ((cx - frame_center[0]) ** 2 + (cy - frame_center[1]) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_box  = (x1, y1, x2 - x1, y2 - y1)
    return best_box


def crop_center_square(frame, size=640):
    h, w    = frame.shape[:2]
    x_start = (w - size) // 2
    y_start = (h - size) // 2
    return frame[y_start:y_start + size, x_start:x_start + size]


# ── Configuration ─────────────────────────────────────────────────────────────

YOLO_MODEL        = 'yolo11s.pt'
LIGHTTRACK_MODEL  = 'lighttrack_cerberus.onnx'
CLASSES_TO_DETECT = [14]                # 14 = bird in COCO
RESOLUTION        = (1280, 720)
CROP_SIZE         = 640

YOLO_CONF         = 0.25

# ── PSR threshold ─────────────────────────────────────────────────────────────
# Unlike raw sigmoid, PSR measures peak *sharpness* relative to the rest of
# the response map.  A flat map (target absent) gives low PSR even when the
# raw peak value looks fine.  This is the primary lever for presence detection.
#
#   < 7   → target likely gone         → reacquire
#   7–10  → marginal / partial occlusion
#   > 10  → solid lock
#
# Raise PSR_THRESH if you still get false locks; lower it if it drops out on
# fast motion or partial occlusion.

PSR_THRESH        = 2 #normall 5, testing 0.1

# ── Search window ─────────────────────────────────────────────────────────────
SEARCH_SCALE_INIT  = 2.0    # template_size_px × this  (first acquire)
SEARCH_SCALE_TRACK = 2.5    # max(tw, th) × this       (adaptive component)
SEARCH_FLOOR_SCALE = 2.75   # template_size_px × this  (floor — prevents collapse)
MIN_SEARCH_PX      = 64     # hard absolute floor


# ── State ─────────────────────────────────────────────────────────────────────

class TrackState:
    SEARCHING   = "SEARCHING"
    TRACKING    = "TRACKING"
    REACQUIRING = "REACQUIRING"


# ── Load models ───────────────────────────────────────────────────────────────

print(f"Loading YOLO {YOLO_MODEL} ...")
yolo_model = YOLO(YOLO_MODEL)
yolo_model.to(0)

print(f"Loading LightTrack {LIGHTTRACK_MODEL} ...")
lt_sess = ort.InferenceSession(LIGHTTRACK_MODEL,
                                providers=["CUDAExecutionProvider",
                                           "CPUExecutionProvider"])
print(f"LightTrack providers: {lt_sess.get_providers()}\n")


# ── Start camera ──────────────────────────────────────────────────────────────

print("Starting camera... Press 'q' to quit, 'r' to reset lock\n")
cap = ThreadedCamera(0, width=RESOLUTION[0], height=RESOLUTION[1]).start()
time.sleep(0.5)

frame_center     = (CROP_SIZE // 2, CROP_SIZE // 2)

state            = TrackState.SEARCHING
template_tensor  = None
template_size_px = 0
search_size_px   = 0
search_origin    = (0, 0)
track_bbox       = None
last_psr         = 0.0
last_raw_conf    = 0.0
last_score_map   = None

fps_start   = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = crop_center_square(frame, CROP_SIZE)

    inf_time = 0.0
    lt_time  = 0.0

    # ── State machine ─────────────────────────────────────────────────────────

    if state in (TrackState.SEARCHING, TrackState.REACQUIRING):
        t0       = time.time()
        results  = yolo_model(frame, verbose=False, conf=YOLO_CONF,
                               classes=CLASSES_TO_DETECT)
        inf_time = (time.time() - t0) * 1000

        bbox = best_detection(results, frame_center, CLASSES_TO_DETECT)

        if bbox is not None:
            bx, by, bw, bh = bbox
            bcx, bcy       = bx + bw // 2, by + bh // 2

            # Tight template: min(bw, bh) crops inside the detection box,
            # minimising background bleed into the template embedding
            template_size_px = min(bw, bh)
            tmpl_crop, _     = extract_square_crop(frame, bcx, bcy,
                                                   template_size_px)
            template_tensor  = preprocess(tmpl_crop, TEMPLATE_SIZE)

            search_size_px = max(int(template_size_px * SEARCH_SCALE_INIT),
                                 MIN_SEARCH_PX)
            search_crop, search_origin = extract_square_crop(frame, bcx, bcy,
                                                              search_size_px)

            t0 = time.time()
            best_box, last_psr, last_raw_conf, last_score_map = run_lighttrack(
                lt_sess, template_tensor, search_crop)
            lt_time = (time.time() - t0) * 1000

            if last_psr >= PSR_THRESH:
                track_bbox = search_box_to_frame(best_box, search_origin,
                                                 search_size_px)
                state = TrackState.TRACKING
                print(f"Target acquired | PSR={last_psr:.1f} | "
                      f"raw={last_raw_conf:.3f} | "
                      f"template_px={template_size_px} | bbox={track_bbox}")
            else:
                print(f"YOLO found target but PSR too low ({last_psr:.1f}) — retrying")

    elif state == TrackState.TRACKING:
        if track_bbox is not None:
            tx, ty, tw, th = track_bbox
            tcx = tx + tw // 2
            tcy = ty + th // 2

            adaptive_size  = int(max(tw, th) * SEARCH_SCALE_TRACK)
            floor_size     = int(template_size_px * SEARCH_FLOOR_SCALE)
            search_size_px = max(adaptive_size, floor_size, MIN_SEARCH_PX)

            search_crop, search_origin = extract_square_crop(frame, tcx, tcy,
                                                              search_size_px)

            t0 = time.time()
            best_box, last_psr, last_raw_conf, last_score_map = run_lighttrack(
                lt_sess, template_tensor, search_crop)
            lt_time = (time.time() - t0) * 1000

            if last_psr >= PSR_THRESH:
                track_bbox = search_box_to_frame(best_box, search_origin,
                                                 search_size_px)
            else:
                print(f"PSR dropped to {last_psr:.1f} — reacquiring")
                state      = TrackState.REACQUIRING
                track_bbox = None

    # ── Draw ──────────────────────────────────────────────────────────────────

    annotated = frame.copy()

    border_color = {
        TrackState.SEARCHING:   (128, 128, 128),
        TrackState.TRACKING:    (0, 255, 0),
        TrackState.REACQUIRING: (0, 100, 255),
    }[state]
    cv2.rectangle(annotated, (0, 0), (CROP_SIZE - 1, CROP_SIZE - 1),
                  border_color, 4)

    # ── Heatmap overlay ───────────────────────────────────────────────────────
    if state == TrackState.TRACKING and last_score_map is not None:
        hmap_small  = (last_score_map * 255).astype(np.uint8)
        hmap_colour = cv2.applyColorMap(
            cv2.resize(hmap_small, (search_size_px, search_size_px),
                       interpolation=cv2.INTER_LINEAR),
            cv2.COLORMAP_JET)

        ox, oy = search_origin
        x1c = max(0, ox);           y1c = max(0, oy)
        x2c = min(CROP_SIZE, ox + search_size_px)
        y2c = min(CROP_SIZE, oy + search_size_px)
        hx1 = x1c - ox;  hy1 = y1c - oy
        hx2 = hx1 + (x2c - x1c)
        hy2 = hy1 + (y2c - y1c)

        roi = annotated[y1c:y2c, x1c:x2c]
        annotated[y1c:y2c, x1c:x2c] = cv2.addWeighted(
            roi, 0.6,
            hmap_colour[hy1:hy2, hx1:hx2], 0.4, 0)

    # ── Search window box ─────────────────────────────────────────────────────
    if state == TrackState.TRACKING:
        ox, oy = search_origin
        cv2.rectangle(annotated,
                      (max(0, ox), max(0, oy)),
                      (min(CROP_SIZE - 1, ox + search_size_px),
                       min(CROP_SIZE - 1, oy + search_size_px)),
                      (200, 200, 0), 1)

    # ── Tracked bbox ──────────────────────────────────────────────────────────
    if track_bbox is not None:
        x, y, w, h = track_bbox
        cx, cy      = x + w // 2, y + h // 2

        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.drawMarker(annotated, (cx, cy), (0, 255, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        cv2.line(annotated, frame_center, (cx, cy), (0, 200, 255), 1)
        cv2.putText(annotated, f"LOCKED  PSR:{last_psr:.1f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # ── Frame centre crosshair ────────────────────────────────────────────────
    cv2.drawMarker(annotated, frame_center, (128, 128, 128),
                   markerType=cv2.MARKER_CROSS, markerSize=30, thickness=1)

    # ── Gimbal error signal ───────────────────────────────────────────────────
    error_x = error_y = 0
    if track_bbox is not None:
        cx = track_bbox[0] + track_bbox[2] // 2
        cy = track_bbox[1] + track_bbox[3] // 2
        error_x = cx - frame_center[0]
        error_y = cy - frame_center[1]

    # ── Stats + PSR bar ───────────────────────────────────────────────────────
    frame_count += 1
    fps = frame_count / (time.time() - fps_start)

    # PSR bar at bottom — makes threshold tuning much easier visually
    psr_clamped = min(max(last_psr, 0.0), 20.0)
    bar_w       = int((psr_clamped / 20.0) * 150)
    bar_color   = (0, 255, 0) if last_psr >= PSR_THRESH else (0, 100, 255)
    cv2.rectangle(annotated, (10, CROP_SIZE - 30),
                  (160, CROP_SIZE - 15), (50, 50, 50), -1)
    if bar_w > 0:
        cv2.rectangle(annotated, (10, CROP_SIZE - 30),
                      (10 + bar_w, CROP_SIZE - 15), bar_color, -1)
    cv2.putText(annotated, f"PSR {last_psr:.1f} / {PSR_THRESH}",
                (165, CROP_SIZE - 17), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, bar_color, 1)

    stats = [
        (f'FPS: {fps:.1f}',                               (0, 255, 0)),
        (f'State: {state}',                               border_color),
        (f'Resolution: {CROP_SIZE}x{CROP_SIZE}',          (255, 255, 0)),
        (f'YOLO:       {inf_time:.1f}ms',                 (255, 100, 100)),
        (f'LightTrack: {lt_time:.1f}ms',                  (255, 100, 100)),
        (f'PSR:        {last_psr:.2f}  (thresh {PSR_THRESH})', bar_color),
        (f'Raw conf:   {last_raw_conf:.3f}',              (150, 150, 200)),
        (f'Template px:{template_size_px}',               (200, 200, 255)),
        (f'Search px:  {search_size_px}',                 (200, 200, 255)),
        (f'Error X: {error_x:+d}px  Y: {error_y:+d}px', (200, 200, 255)),
    ]
    for i, (text, color) in enumerate(stats):
        cv2.putText(annotated, text, (10, 30 + i * 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    cv2.imshow('Goose Tracker (YOLO + LightTrack)', annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        state           = TrackState.SEARCHING
        track_bbox      = None
        template_tensor = None
        print("Lock reset — searching")

cap.stop()
cv2.destroyAllWindows()
print(f"\nFinal Average FPS: {fps:.1f}")