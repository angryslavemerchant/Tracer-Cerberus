import cv2
import numpy as np
import hailo_platform as hpf
from pathlib import Path

ROOT         = Path(__file__).resolve().parent.parent
YOLO_HEF     = str(ROOT / "Models" / "yolov8s.hef")
CERBERUS_HEF = str(ROOT / "Models" / "CerberusCoreS.hef")

CLASSES_TO_DETECT = [14]
YOLO_CONF         = 0.5
PSR_THRESH        = 18
CROP_SIZE         = 640
TEMPLATE_SIZE     = 128
SEARCH_SIZE       = 256
STRIDE            = 16
SCORE_SIZE        = SEARCH_SIZE // STRIDE
SEARCH_SCALE_INIT = 2.0
MIN_SEARCH_PX     = 64


class TrackState:
    SEARCHING   = "SEARCHING"
    TRACKING    = "TRACKING"
    REACQUIRING = "REACQUIRING"


def _make_grid(score_size, stride, search_size):
    half = score_size // 2
    xs   = np.arange(score_size) - np.floor(float(half))
    ys   = np.arange(score_size) - np.floor(float(half))
    gx, gy = np.meshgrid(xs, ys)
    return gx * stride + search_size // 2, gy * stride + search_size // 2

_GRID_X, _GRID_Y = _make_grid(SCORE_SIZE, STRIDE, SEARCH_SIZE)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


def _compute_psr(score_map, exclude_radius=2):
    peak_idx       = int(np.argmax(score_map))
    peak_r, peak_c = np.unravel_index(peak_idx, score_map.shape)
    h, w = score_map.shape
    mask = np.ones_like(score_map, dtype=bool)
    mask[max(0, peak_r - exclude_radius):min(h, peak_r + exclude_radius + 1),
         max(0, peak_c - exclude_radius):min(w, peak_c + exclude_radius + 1)] = False
    sidelobe = score_map[mask]
    psr = (score_map[peak_r, peak_c] - sidelobe.mean()) / (sidelobe.std() + 1e-6)
    return float(psr), (peak_r, peak_c)


def _preprocess_yolo(frame_bgr, input_shape):
    h, w = input_shape[:2]
    return np.expand_dims(cv2.resize(frame_bgr, (w, h)), axis=0)


def _preprocess_cerberus(img_bgr, out_size):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return np.expand_dims(cv2.resize(img_rgb, (out_size, out_size)).astype(np.float32), axis=0)


def _extract_square_crop(frame, cx, cy, size):
    h, w   = frame.shape[:2]
    half   = size // 2
    ox, oy = cx - half, cy - half
    x1c, y1c = max(0, ox), max(0, oy)
    x2c, y2c = max(x1c, min(w, ox + size)), max(y1c, min(h, oy + size))
    crop = np.zeros((size, size, 3), dtype=np.uint8)
    rh, rw = y2c - y1c, x2c - x1c
    if rh > 0 and rw > 0:
        crop[y1c - oy:y1c - oy + rh, x1c - ox:x1c - ox + rw] = frame[y1c:y2c, x1c:x2c]
    return crop, (ox, oy)


def _heatmap_center_to_frame(cx_search, cy_search, origin, search_size_px, box_size_px):
    scale    = search_size_px / SEARCH_SIZE
    ox, oy   = origin
    cx_frame = ox + cx_search * scale
    cy_frame = oy + cy_search * scale
    half     = box_size_px // 2
    return int(cx_frame - half), int(cy_frame - half), box_size_px, box_size_px


def _best_detection(per_class_dets, frame_h, frame_w, frame_center):
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


class TrackerHeadless:
    def __init__(self):
        print("[Tracker] Loading models...")
        hef_yolo     = hpf.HEF(YOLO_HEF)
        hef_cerberus = hpf.HEF(CERBERUS_HEF)

        vdevice_params = hpf.VDevice.create_params()
        vdevice_params.scheduling_algorithm = hpf.HailoSchedulingAlgorithm.ROUND_ROBIN
        self._target = hpf.VDevice(vdevice_params)

        ng_yolo = self._target.configure(
            hef_yolo,
            hpf.ConfigureParams.create_from_hef(hef_yolo, interface=hpf.HailoStreamInterface.PCIe)
        )[0]
        ng_cerberus = self._target.configure(
            hef_cerberus,
            hpf.ConfigureParams.create_from_hef(hef_cerberus, interface=hpf.HailoStreamInterface.PCIe)
        )[0]

        self._yolo_input_info    = hef_yolo.get_input_vstream_infos()[0]
        self._yolo_output_info   = hef_yolo.get_output_vstream_infos()[0]
        self._cerberus_input_infos = hef_cerberus.get_input_vstream_infos()
        self._cerberus_output_info = hef_cerberus.get_output_vstream_infos()[0]

        self._yolo_pipeline = hpf.InferVStreams(
            ng_yolo,
            hpf.InputVStreamParams.make_from_network_group(ng_yolo, quantized=True,  format_type=hpf.FormatType.UINT8),
            hpf.OutputVStreamParams.make_from_network_group(ng_yolo, quantized=False, format_type=hpf.FormatType.FLOAT32),
        )
        self._cerberus_pipeline = hpf.InferVStreams(
            ng_cerberus,
            hpf.InputVStreamParams.make_from_network_group(ng_cerberus, quantized=False, format_type=hpf.FormatType.FLOAT32),
            hpf.OutputVStreamParams.make_from_network_group(ng_cerberus, quantized=False, format_type=hpf.FormatType.FLOAT32),
        )
        self._yolo_pipeline.__enter__()
        self._cerberus_pipeline.__enter__()

        self.state            = TrackState.SEARCHING
        self.track_bbox       = None
        self.last_psr         = 0.0
        self.last_score_map   = None
        self.frame_center     = (0, 0)  # set on first frame

        self._template_tensor  = None
        self._template_size_px = 0
        self._search_size_px   = 0
        self._search_origin    = (0, 0)
        self.lock_bbox_h_px    = 0

        print("[Tracker] Ready.")

    def reset(self):
        self.state           = TrackState.SEARCHING
        self.track_bbox      = None
        self._template_tensor = None
        print("[Tracker] Reset.")

    def update(self, frame):
        """
        Run one tracking step on a BGR frame cropped to CROP_SIZE x CROP_SIZE.
        Returns (error_x, error_y, state).
        """
        frame_h, frame_w = frame.shape[:2]
        self.frame_center = (frame_w // 2, frame_h // 2)

        if self.state in (TrackState.SEARCHING, TrackState.REACQUIRING):
            yolo_in = _preprocess_yolo(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), self._yolo_input_info.shape)
            results = self._yolo_pipeline.infer({self._yolo_input_info.name: yolo_in})
            per_class_dets = results[self._yolo_output_info.name][0]

            bbox = _best_detection(per_class_dets, frame_h, frame_w, self.frame_center)
            if bbox is not None:
                bx, by, bw, bh = bbox
                bcx, bcy = bx + bw // 2, by + bh // 2

                self._template_size_px = min(bw, bh)
                tmpl_crop, _           = _extract_square_crop(frame, bcx, bcy, self._template_size_px)
                self._template_tensor  = _preprocess_cerberus(tmpl_crop, TEMPLATE_SIZE)

                self._search_size_px = max(int(self._template_size_px * SEARCH_SCALE_INIT), MIN_SEARCH_PX)
                search_crop, self._search_origin = _extract_square_crop(
                    frame, bcx, bcy, self._search_size_px)

                cx_s, cy_s, self.last_psr, _, self.last_score_map = self._run_cerberus(search_crop)

                if self.last_psr >= PSR_THRESH:
                    self.track_bbox = _heatmap_center_to_frame(
                        cx_s, cy_s, self._search_origin, self._search_size_px, self._template_size_px)
                    self.lock_bbox_h_px = bh
                    self.state = TrackState.TRACKING
                    print(f"[Tracker] Locked | PSR={self.last_psr:.1f}")

        elif self.state == TrackState.TRACKING:
            tx, ty, tw, th = self.track_bbox
            search_crop, self._search_origin = _extract_square_crop(
                frame, tx + tw // 2, ty + th // 2, self._search_size_px)

            cx_s, cy_s, self.last_psr, _, self.last_score_map = self._run_cerberus(search_crop)

            if self.last_psr >= PSR_THRESH:
                self.track_bbox = _heatmap_center_to_frame(
                    cx_s, cy_s, self._search_origin, self._search_size_px, self._template_size_px)
            else:
                print(f"[Tracker] Lost | PSR={self.last_psr:.1f}")
                self.state = TrackState.REACQUIRING
                self.track_bbox = None

        error_x = error_y = 0
        if self.track_bbox is not None:
            error_x = self.track_bbox[0] + self.track_bbox[2] // 2 - self.frame_center[0]
            error_y = self.track_bbox[1] + self.track_bbox[3] // 2 - self.frame_center[1]

        return error_x, error_y, self.state

    def _run_cerberus(self, search_bgr):
        search_tensor = _preprocess_cerberus(search_bgr, SEARCH_SIZE)
        input_data = {}
        for info in self._cerberus_input_infos:
            input_data[info.name] = self._template_tensor if info.shape[0] == TEMPLATE_SIZE else search_tensor
        results = self._cerberus_pipeline.infer(input_data)
        raw     = results[self._cerberus_output_info.name]
        cls_sig = _sigmoid(raw[0, :, :, 0]).astype(np.float32)

        # PSR on original 16x16 map
        psr, (peak_r, peak_c) = _compute_psr(cls_sig)
        raw_conf = float(cls_sig[peak_r, peak_c])

        # Sub-grid peak: upsample to 256x256 and find hottest pixel
        upsampled = cv2.resize(cls_sig, (SEARCH_SIZE, SEARCH_SIZE), interpolation=cv2.INTER_CUBIC)
        flat_idx  = int(np.argmax(upsampled))
        py, px    = np.unravel_index(flat_idx, upsampled.shape)

        return float(px), float(py), psr, raw_conf, cls_sig

    def stop(self):
        self._yolo_pipeline.__exit__(None, None, None)
        self._cerberus_pipeline.__exit__(None, None, None)
        self._target.__exit__(None, None, None)
        print("[Tracker] Stopped.")
