from pathlib import Path

import cv2
import numpy as np


INPUT_SIZE = 416
STRIDES = (8, 16, 32)
PERSON_SCORE_FLOOR = 0.10
NMS_IOU_THRESHOLD = 0.45


def letterbox(image: np.ndarray):
    height, width = image.shape[:2]
    scale = min(INPUT_SIZE / height, INPUT_SIZE / width)
    resized = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))))
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    canvas[:resized.shape[0], :resized.shape[1]] = resized
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32), scale


def anchor_grid():
    grids = []
    expanded_strides = []
    for stride in STRIDES:
        side = INPUT_SIZE // stride
        xs, ys = np.meshgrid(np.arange(side), np.arange(side))
        grids.append(np.stack((xs.ravel(), ys.ravel()), axis=1))
        expanded_strides.append(np.full((side * side, 1), stride))
    return np.concatenate(grids), np.concatenate(expanded_strides)


def suppress_overlaps(boxes, scores):
    xywh = boxes.copy()
    xywh[:, 2:] -= xywh[:, :2]
    indices = cv2.dnn.NMSBoxes(
        xywh.tolist(), scores.tolist(), PERSON_SCORE_FLOOR, NMS_IOU_THRESHOLD,
    )
    return np.asarray(indices, dtype=np.intp).reshape(-1)


class PersonDetector:
    def __init__(self, model_path: Path, threads: int):
        import onnxruntime

        if not model_path.is_file():
            raise FileNotFoundError(f"Missing model {model_path}. Run download_model.py first.")
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"],
        )
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 3, INPUT_SIZE, INPUT_SIZE]:
            raise ValueError("Expected the official YOLOX-Nano 416x416 ONNX model")
        self.input_name = inputs[0].name
        self.grid, self.strides = anchor_grid()

    def detect(self, bgr: np.ndarray) -> np.ndarray:
        tensor, scale = letterbox(bgr)
        model_output = self.session.run(None, {self.input_name: tensor})[0]
        if model_output.shape != (1, len(self.grid), 85):
            raise ValueError(f"Unexpected YOLOX output shape: {model_output.shape}")

        prediction = model_output[0]
        # COCO person is class zero. Preserve low-score boxes for ByteTrack's second match.
        scores = prediction[:, 4] * prediction[:, 5]
        keep = np.isfinite(scores) & (scores > PERSON_SCORE_FLOOR)
        raw_boxes = prediction[keep, :4]
        scores = scores[keep]
        if not len(raw_boxes):
            return np.empty((0, 5), dtype=np.float32)

        strides = self.strides[keep]
        centers = (raw_boxes[:, :2] + self.grid[keep]) * strides
        with np.errstate(over="ignore"):
            half_sizes = np.exp(raw_boxes[:, 2:]) * strides / 2
        boxes = np.concatenate((centers - half_sizes, centers + half_sizes), axis=1) / scale
        valid = np.isfinite(boxes).all(axis=1)
        height, width = bgr.shape[:2]
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, width)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, height)
        valid &= (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid]
        scores = scores[valid]
        if not len(boxes):
            return np.empty((0, 5), dtype=np.float32)

        keep = suppress_overlaps(boxes, scores)
        return np.column_stack((boxes[keep], scores[keep])).astype(np.float32)
