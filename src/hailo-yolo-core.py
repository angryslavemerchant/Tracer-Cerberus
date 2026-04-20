import cv2
import numpy as np
import hailo_platform as hpf
from picamera2 import Picamera2
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent.parent
HEF_PATH = str(ROOT / "Models" / "yolov8s.hef")
CONF_THRESHOLD = 0.5

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

COLORS = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3), dtype=np.uint8)


def preprocess(frame, input_shape):
    h, w = input_shape[:2]
    return np.expand_dims(cv2.resize(frame, (w, h)).astype(np.float32), axis=0)


def draw_detections(frame, per_class_dets, orig_h, orig_w):
    for cls_idx, arr in enumerate(per_class_dets):
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.shape[0] == 5 and arr.shape[1] != 5:
            arr = arr.T
        for y1, x1, y2, x2, score in arr:
            if score < CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = int(x1 * orig_w), int(y1 * orig_h), int(x2 * orig_w), int(y2 * orig_h)
            color = COLORS[cls_idx].tolist()
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{COCO_CLASSES[cls_idx]} {score:.2f}", (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame


def main():
    hef = hpf.HEF(HEF_PATH)

    with hpf.VDevice() as target:
        configure_params = hpf.ConfigureParams.create_from_hef(hef, interface=hpf.HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()

        input_info = hef.get_input_vstream_infos()[0]
        output_info = hef.get_output_vstream_infos()[0]

        input_params = hpf.InputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)
        output_params = hpf.OutputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)

        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(main={"size": (1280, 720), "format": "RGB888"}))
        picam2.start()

        with network_group.activate(network_group_params):
            with hpf.InferVStreams(network_group, input_params, output_params) as pipeline:
                prev_time = time.time()
                while True:
                    frame = picam2.capture_array()
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                    orig_h, orig_w = frame.shape[:2]

                    # Feed BGR→RGB to model (correct channel order for YOLOv8)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pipeline.infer({input_info.name: preprocess(frame_rgb, input_info.shape)})
                    per_class_dets = results[output_info.name][0]

                    now = time.time()
                    fps = 1.0 / (now - prev_time)
                    prev_time = now

                    # Display raw frame directly (picamera2 RGB888 outputs BGR on this system)
                    annotated = draw_detections(frame.copy(), per_class_dets, orig_h, orig_w)
                    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow("YOLOv8s - Hailo", annotated)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
