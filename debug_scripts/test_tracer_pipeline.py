import cv2
import numpy as np
import hailo_platform as hpf
from picamera2 import Picamera2
from libcamera import Transform
from pathlib import Path
import time
import threading

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT          = Path(__file__).resolve().parent.parent
YOLO_HEF      = str(ROOT / "Models" / "yolov8s.hef")
CERBERUS_HEF  = str(ROOT / "Models" / "CerberusCoreS.hef")

# ── Config ────────────────────────────────────────────────────────────────────

CLASSES_TO_DETECT = [14]      # 14 = bird
YOLO_CONF         = 0.5
PSR_THRESH        = 18
CROP_SIZE         = 640
TEMPLATE_SIZE     = 128
SEARCH_SIZE       = 256
STRIDE            = 16
SCORE_SIZE        = SEARCH_SIZE // STRIDE   # 16
SEARCH_SCALE_INIT = 2.0
MIN_SEARCH_PX     = 64


# ── COCO classes (YOLO output index) ─────────────────────────────────────────

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

# ── State machine ─────────────────────────────────────────────────────────────

class TrackState:
    SEARCHING   = "SEARCHING"
    TRACKING    = "TRACKING"
    REACQUIRING = "REACQUIRING"

# ── Grid (precomputed) ────────────────────────────────────────────────────────

def _make_grid(score_size, stride, search_size):
    half = score_size // 2
    xs   = np.arange(score_size) - np.floor(float(half))
    ys   = np.arange(score_size) - np.floor(float(half))
    gx, gy = np.meshgrid(xs, ys)
    return gx * stride + search_size // 2, gy * stride + search_size // 2

_GRID_X, _GRID_Y = _make_grid(SCORE_SIZE, STRIDE, SEARCH_SIZE)

# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_yolo(frame_bgr, input_shape):
    """Resize to model input size, keep uint8 NHWC."""
    h, w = input_shape[:2]
    return np.expand_dims(cv2.resize(frame_bgr, (w, h)), axis=0)


def preprocess_cerberus(img_bgr, out_size):
    """BGR uint8 → float32 NHWC [0,255]. Normalization is baked into the HEF."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = cv2.resize(img_rgb, (out_size, out_size)).astype(np.float32)
    return x[np.newaxis]   # (1, H, W, 3)

# ── PSR ───────────────────────────────────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


def compute_psr(score_map, exclude_radius=2):
    peak_idx       = int(np.argmax(score_map))
    peak_r, peak_c = np.unravel_index(peak_idx, score_map.shape)
    h, w = score_map.shape
    mask = np.ones_like(score_map, dtype=bool)
    mask[max(0, peak_r - exclude_radius):min(h, peak_r + exclude_radius + 1),
         max(0, peak_c - exclude_radius):min(w, peak_c + exclude_radius + 1)] = False
    sidelobe = score_map[mask]
    psr = (score_map[peak_r, peak_c] - sidelobe.mean()) / (sidelobe.std() + 1e-6)
    return float(psr), (peak_r, peak_c)

# ── Cerberus inference ────────────────────────────────────────────────────────

def run_cerberus(pipeline, cerberus_input_infos, cerberus_output_info,
                 template_tensor, search_bgr):
    search_tensor = preprocess_cerberus(search_bgr, SEARCH_SIZE)

    input_data = {}
    for info in cerberus_input_infos:
        if info.shape[0] == TEMPLATE_SIZE:
            input_data[info.name] = template_tensor
        else:
            input_data[info.name] = search_tensor

    results  = pipeline.infer(input_data)
    raw      = results[cerberus_output_info.name]   # (1, 16, 16, 1)
    cls_sig  = sigmoid(raw[0, :, :, 0]).astype(np.float32)  # (16, 16)

    psr, (peak_r, peak_c) = compute_psr(cls_sig)
    raw_conf  = float(cls_sig[peak_r, peak_c])
    cx_search = float(_GRID_X[peak_r, peak_c])
    cy_search = float(_GRID_Y[peak_r, peak_c])
    return cx_search, cy_search, psr, raw_conf, cls_sig

# ── YOLO helpers ──────────────────────────────────────────────────────────────

def best_detection_hailo(per_class_dets, frame_h, frame_w, frame_center):
    best_box, best_dist = None, float('inf')
    for cls_idx in CLASSES_TO_DETECT:
        arr = per_class_dets[cls_idx]
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.shape[0] == 5 and arr.shape[1] != 5:
            arr = arr.T
        for y1, x1, y2, x2, score in arr:
            if score < YOLO_CONF:
                continue
            x1i, y1i = int(x1 * frame_w), int(y1 * frame_h)
            x2i, y2i = int(x2 * frame_w), int(y2 * frame_h)
            cx, cy = (x1i + x2i) // 2, (y1i + y2i) // 2
            dist = ((cx - frame_center[0])**2 + (cy - frame_center[1])**2)**0.5
            if dist < best_dist:
                best_dist = dist
                best_box  = (x1i, y1i, x2i - x1i, y2i - y1i)
    return best_box

# ── Crop helpers ──────────────────────────────────────────────────────────────

def crop_center_square(frame, size=640):
    h, w = frame.shape[:2]
    return frame[(h - size) // 2:(h + size) // 2,
                 (w - size) // 2:(w + size) // 2]


def extract_square_crop(frame, cx, cy, size):
    h, w   = frame.shape[:2]
    half   = size // 2
    ox, oy = cx - half, cy - half
    x1c, y1c = max(0, ox),          max(0, oy)
    x2c, y2c = max(x1c, min(w, ox + size)), max(y1c, min(h, oy + size))
    crop = np.zeros((size, size, 3), dtype=np.uint8)
    rh, rw = y2c - y1c, x2c - x1c
    if rh > 0 and rw > 0:
        crop[y1c - oy:y1c - oy + rh, x1c - ox:x1c - ox + rw] = frame[y1c:y2c, x1c:x2c]
    return crop, (ox, oy)


def heatmap_center_to_frame(cx_search, cy_search, origin, search_size_px, box_size_px):
    scale    = search_size_px / SEARCH_SIZE
    ox, oy   = origin
    cx_frame = ox + cx_search * scale
    cy_frame = oy + cy_search * scale
    half     = box_size_px // 2
    return int(cx_frame - half), int(cy_frame - half), box_size_px, box_size_px

# ── Display thread ────────────────────────────────────────────────────────────

def display_thread(shared, lock, stop_event):
    while not stop_event.is_set():
        with lock:
            frame = shared["frame"]
        if frame is not None:
            cv2.imshow("Tracer - Cerberus", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
        elif cv2.waitKey(1) & 0xFF == ord("r"):
            with lock:
                shared["reset"] = True
    cv2.destroyAllWindows()

# ── Main ──────────────────────────────────────────────────────────────────────

def draw_frame(frame, state, track_bbox, score_map, search_origin, search_size_px,
               psr, frame_center, fps, yolo_ms, cerberus_ms, template_size_px):
    annotated    = frame.copy()
    border_color = {
        TrackState.SEARCHING:   (128, 128, 128),
        TrackState.TRACKING:    (0, 255, 0),
        TrackState.REACQUIRING: (0, 100, 255),
    }[state]

    cv2.rectangle(annotated, (0, 0), (CROP_SIZE - 1, CROP_SIZE - 1), border_color, 4)

    if state == TrackState.TRACKING and score_map is not None:
        hmap = cv2.applyColorMap(
            cv2.resize((score_map * 255).astype(np.uint8),
                       (search_size_px, search_size_px), interpolation=cv2.INTER_LINEAR),
            cv2.COLORMAP_JET)
        ox, oy = search_origin
        x1c, y1c = max(0, ox), max(0, oy)
        x2c = max(x1c, min(CROP_SIZE, ox + search_size_px))
        y2c = max(y1c, min(CROP_SIZE, oy + search_size_px))
        hx1, hy1 = x1c - ox, y1c - oy
        hx2, hy2 = hx1 + (x2c - x1c), hy1 + (y2c - y1c)
        if y2c > y1c and x2c > x1c:
            annotated[y1c:y2c, x1c:x2c] = cv2.addWeighted(
                annotated[y1c:y2c, x1c:x2c], 0.6, hmap[hy1:hy2, hx1:hx2], 0.4, 0)
        ox, oy = search_origin
        cv2.rectangle(annotated,
                      (max(0, ox), max(0, oy)),
                      (min(CROP_SIZE - 1, ox + search_size_px),
                       min(CROP_SIZE - 1, oy + search_size_px)),
                      (200, 200, 0), 1)

    if track_bbox is not None:
        x, y, w, h = track_bbox
        cx, cy = x + w // 2, y + h // 2
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.drawMarker(annotated, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.line(annotated, frame_center, (cx, cy), (0, 200, 255), 1)
        cv2.putText(annotated, f"LOCKED  PSR:{psr:.1f}", (x, max(y - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.drawMarker(annotated, frame_center, (128, 128, 128), cv2.MARKER_CROSS, 30, 1)

    error_x = error_y = 0
    if track_bbox is not None:
        error_x = track_bbox[0] + track_bbox[2] // 2 - frame_center[0]
        error_y = track_bbox[1] + track_bbox[3] // 2 - frame_center[1]

    psr_safe  = psr if psr == psr else 0.0
    bar_w     = int((min(max(psr_safe, 0.0), 20.0) / 20.0) * 150)
    bar_color = (0, 255, 0) if psr >= PSR_THRESH else (0, 100, 255)
    cv2.rectangle(annotated, (10, CROP_SIZE - 30), (160, CROP_SIZE - 15), (50, 50, 50), -1)
    if bar_w > 0:
        cv2.rectangle(annotated, (10, CROP_SIZE - 30), (10 + bar_w, CROP_SIZE - 15), bar_color, -1)
    cv2.putText(annotated, f"PSR {psr:.1f}/{PSR_THRESH}",
                (165, CROP_SIZE - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bar_color, 1)

    for i, (text, color) in enumerate([
        (f"FPS: {fps:.1f}",                            (0, 255, 0)),
        (f"State: {state}",                            border_color),
        (f"YOLO:      {yolo_ms:.1f}ms",               (255, 100, 100)),
        (f"Cerberus:  {cerberus_ms:.1f}ms",           (255, 100, 100)),
        (f"PSR:       {psr:.2f} (t={PSR_THRESH})",    bar_color),
        (f"Template:  {template_size_px}px",           (200, 200, 255)),
        (f"Search:    {search_size_px}px",             (200, 200, 255)),
        (f"Error X: {error_x:+d}  Y: {error_y:+d}",  (200, 200, 255)),
    ]):
        cv2.putText(annotated, text, (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return annotated


def main():
    print("[INIT] Loading HEFs...")
    hef_yolo     = hpf.HEF(YOLO_HEF)
    hef_cerberus = hpf.HEF(CERBERUS_HEF)

    vdevice_params = hpf.VDevice.create_params()
    vdevice_params.scheduling_algorithm = hpf.HailoSchedulingAlgorithm.ROUND_ROBIN

    print("[INIT] Opening VDevice (ROUND_ROBIN)")
    with hpf.VDevice(vdevice_params) as target:
        ng_yolo = target.configure(
            hef_yolo,
            hpf.ConfigureParams.create_from_hef(hef_yolo, interface=hpf.HailoStreamInterface.PCIe)
        )[0]
        ng_cerberus = target.configure(
            hef_cerberus,
            hpf.ConfigureParams.create_from_hef(hef_cerberus, interface=hpf.HailoStreamInterface.PCIe)
        )[0]

        yolo_input_info  = hef_yolo.get_input_vstream_infos()[0]
        yolo_output_info = hef_yolo.get_output_vstream_infos()[0]

        cerberus_input_infos = hef_cerberus.get_input_vstream_infos()
        cerberus_output_info = hef_cerberus.get_output_vstream_infos()[0]

        yolo_in_params  = hpf.InputVStreamParams.make_from_network_group(
            ng_yolo, quantized=True, format_type=hpf.FormatType.UINT8)
        yolo_out_params = hpf.OutputVStreamParams.make_from_network_group(
            ng_yolo, quantized=False, format_type=hpf.FormatType.FLOAT32)

        cerberus_in_params  = hpf.InputVStreamParams.make_from_network_group(
            ng_cerberus, quantized=False, format_type=hpf.FormatType.FLOAT32)
        cerberus_out_params = hpf.OutputVStreamParams.make_from_network_group(
            ng_cerberus, quantized=False, format_type=hpf.FormatType.FLOAT32)

        print("[INIT] Starting camera...")
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(
            main={"size": (1280, 720), "format": "RGB888"},
            controls={"FrameRate": 120},
            transform=Transform(hflip=True, vflip=True)
        ))
        picam2.start()

        shared     = {"frame": None, "reset": False}
        lock       = threading.Lock()
        stop_event = threading.Event()
        t = threading.Thread(target=display_thread, args=(shared, lock, stop_event), daemon=True)
        t.start()

        with hpf.InferVStreams(ng_yolo, yolo_in_params, yolo_out_params) as yolo_pipeline:
            with hpf.InferVStreams(ng_cerberus, cerberus_in_params, cerberus_out_params) as cerberus_pipeline:

                state            = TrackState.SEARCHING
                template_tensor  = None
                template_size_px = 0
                search_size_px   = 0
                search_origin    = (0, 0)
                track_bbox       = None
                last_psr         = 0.0
                last_raw_conf    = 0.0
                last_score_map   = None
                frame_center     = (CROP_SIZE // 2, CROP_SIZE // 2)

                prev_time = time.time()

                print("[LOOP] Running - press 'q' to quit, 'r' to reset lock")
                while not stop_event.is_set():
                    raw_frame = picam2.capture_array()
                    frame     = crop_center_square(raw_frame, CROP_SIZE)

                    yolo_time = lt_time = 0.0

                    # ── State machine ──────────────────────────────────────────
                    if state in (TrackState.SEARCHING, TrackState.REACQUIRING):
                        t0 = time.time()
                        yolo_input = preprocess_yolo(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), yolo_input_info.shape)
                        results = yolo_pipeline.infer({yolo_input_info.name: yolo_input})
                        per_class_dets = results[yolo_output_info.name][0]
                        yolo_time = (time.time() - t0) * 1000

                        bbox = best_detection_hailo(per_class_dets, CROP_SIZE, CROP_SIZE, frame_center)

                        if bbox is not None:
                            bx, by, bw, bh = bbox
                            bcx, bcy = bx + bw // 2, by + bh // 2

                            template_size_px = min(bw, bh)
                            tmpl_crop, _     = extract_square_crop(frame, bcx, bcy, template_size_px)
                            template_tensor  = preprocess_cerberus(tmpl_crop, TEMPLATE_SIZE)

                            search_size_px = max(int(template_size_px * SEARCH_SCALE_INIT), MIN_SEARCH_PX)
                            search_crop, search_origin = extract_square_crop(frame, bcx, bcy, search_size_px)

                            t0 = time.time()
                            cx_s, cy_s, last_psr, last_raw_conf, last_score_map = run_cerberus(
                                cerberus_pipeline, cerberus_input_infos, cerberus_output_info,
                                template_tensor, search_crop)
                            lt_time = (time.time() - t0) * 1000

                            if last_psr >= PSR_THRESH:
                                track_bbox = heatmap_center_to_frame(
                                    cx_s, cy_s, search_origin, search_size_px, template_size_px)
                                state = TrackState.TRACKING
                                print(f"[LOCK] PSR={last_psr:.1f} template_px={template_size_px} bbox={track_bbox}")
                            else:
                                print(f"[SEARCH] YOLO found target but PSR too low ({last_psr:.1f})")

                    elif state == TrackState.TRACKING:
                        tx, ty, tw, th = track_bbox
                        tcx, tcy = tx + tw // 2, ty + th // 2
                        search_crop, search_origin = extract_square_crop(frame, tcx, tcy, search_size_px)

                        t0 = time.time()
                        cx_s, cy_s, last_psr, last_raw_conf, last_score_map = run_cerberus(
                            cerberus_pipeline, cerberus_input_infos, cerberus_output_info,
                            template_tensor, search_crop)
                        lt_time = (time.time() - t0) * 1000

                        if last_psr >= PSR_THRESH:
                            track_bbox = heatmap_center_to_frame(
                                cx_s, cy_s, search_origin, search_size_px, template_size_px)
                        else:
                            print(f"[LOST] PSR dropped to {last_psr:.1f} - reacquiring")
                            state = TrackState.REACQUIRING
                            track_bbox = None

                    now = time.time()
                    fps = 1.0 / (now - prev_time)
                    prev_time = now

                    annotated = draw_frame(
                        frame, state, track_bbox, last_score_map,
                        search_origin, search_size_px, last_psr,
                        frame_center, fps, yolo_time, lt_time, template_size_px
                    )

                    with lock:
                        shared["frame"] = annotated
                        if shared["reset"]:
                            state = TrackState.SEARCHING
                            track_bbox = template_tensor = None
                            shared["reset"] = False
                            print("[RESET] Searching...")

        picam2.stop()
        t.join()


if __name__ == "__main__":
    main()
