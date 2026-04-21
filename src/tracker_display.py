import cv2
import numpy as np
import threading
from tracker import TrackerHeadless, TrackState, PSR_THRESH


class TrackerDisplay(TrackerHeadless):
    def __init__(self):
        super().__init__()
        self._lock       = threading.Lock()
        self._stop_event = threading.Event()
        self._shared     = {"frame": None, "state": None, "reset": False}
        self._thread     = threading.Thread(target=self._display_loop, daemon=True)
        self.last_key    = -1
        self._thread.start()

    def update(self, frame):
        error_x, error_y, state = super().update(frame)

        # Hand raw frame + state snapshot to display thread — no drawing here
        with self._lock:
            self._shared["frame"] = frame
            self._shared["state"] = self._snapshot()
            if self._shared["reset"]:
                self.reset()
                self._shared["reset"] = False

        return error_x, error_y, state

    def should_stop(self):
        return self._stop_event.is_set()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        super().stop()

    def _snapshot(self):
        return {
            "state":            self.state,
            "track_bbox":       self.track_bbox,
            "score_map":        self.last_score_map,
            "psr":              self.last_psr,
            "search_origin":    self._search_origin,
            "search_size_px":   self._search_size_px,
            "template_size_px": self._template_size_px,
            "frame_center":     self.frame_center,
        }

    def _display_loop(self):
        while not self._stop_event.is_set():
            with self._lock:
                frame    = self._shared["frame"]
                snapshot = self._shared["state"]

            if frame is not None and snapshot is not None:
                cv2.imshow("Tracer - Cerberus", self._draw(frame, snapshot))

            key = cv2.waitKey(1) & 0xFF
            self.last_key = key
            if key == ord("q"):
                self._stop_event.set()
            elif key == ord("r"):
                with self._lock:
                    self._shared["reset"] = True

        cv2.destroyAllWindows()

    def _draw(self, frame, s):
        annotated = frame.copy()
        fh, fw    = frame.shape[:2]

        state            = s["state"]
        track_bbox       = s["track_bbox"]
        score_map        = s["score_map"]
        psr              = s["psr"]
        search_origin    = s["search_origin"]
        search_size_px   = s["search_size_px"]
        template_size_px = s["template_size_px"]
        frame_center     = s["frame_center"]

        border_color = {
            TrackState.SEARCHING:   (128, 128, 128),
            TrackState.TRACKING:    (0, 255, 0),
            TrackState.REACQUIRING: (0, 100, 255),
        }[state]

        cv2.rectangle(annotated, (0, 0), (fw - 1, fh - 1), border_color, 4)

        if state == TrackState.TRACKING and score_map is not None:
            hmap = cv2.applyColorMap(
                cv2.resize((score_map * 255).astype(np.uint8),
                           (search_size_px, search_size_px), interpolation=cv2.INTER_LINEAR),
                cv2.COLORMAP_JET)
            ox, oy = search_origin
            x1c, y1c = max(0, ox), max(0, oy)
            x2c = max(x1c, min(fw, ox + search_size_px))
            y2c = max(y1c, min(fh, oy + search_size_px))
            hx1, hy1 = x1c - ox, y1c - oy
            hx2, hy2 = hx1 + (x2c - x1c), hy1 + (y2c - y1c)
            if y2c > y1c and x2c > x1c:
                annotated[y1c:y2c, x1c:x2c] = cv2.addWeighted(
                    annotated[y1c:y2c, x1c:x2c], 0.6, hmap[hy1:hy2, hx1:hx2], 0.4, 0)
            cv2.rectangle(annotated,
                          (max(0, ox), max(0, oy)),
                          (min(fw - 1, ox + search_size_px), min(fh - 1, oy + search_size_px)),
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
        cv2.rectangle(annotated, (10, fh - 30), (160, fh - 15), (50, 50, 50), -1)
        if bar_w > 0:
            cv2.rectangle(annotated, (10, fh - 30), (10 + bar_w, fh - 15), bar_color, -1)
        cv2.putText(annotated, f"PSR {psr:.1f}/{PSR_THRESH}",
                    (165, fh - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bar_color, 1)

        for i, (text, color) in enumerate([
            (f"State: {state}",                            border_color),
            (f"PSR:       {psr:.2f} (t={PSR_THRESH})",    bar_color),
            (f"Template:  {template_size_px}px",           (200, 200, 255)),
            (f"Search:    {search_size_px}px",             (200, 200, 255)),
            (f"Error X: {error_x:+d}  Y: {error_y:+d}",  (200, 200, 255)),
        ]):
            cv2.putText(annotated, text, (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return annotated
