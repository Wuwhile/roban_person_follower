"""IoU matching for the vendored tracker using NumPy and SciPy."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def linear_assignment(cost, thresh):
    rows, cols = cost.shape
    if not rows or not cols:
        return [], tuple(range(rows)), tuple(range(cols))

    # Dummy columns let a track stay unmatched. Subtracting the cost limit
    # from real matches yields the same cost as charging to leave a pair unmatched.
    extended = np.zeros((rows, cols + rows), dtype=float)
    extended[:, :cols] = np.where(cost <= thresh, cost - thresh - 1e-12, np.inf)
    row_indices, col_indices = linear_sum_assignment(extended)
    matches = [(row, col) for row, col in zip(row_indices, col_indices) if col < cols]
    matched_rows = {row for row, _ in matches}
    matched_cols = {col for _, col in matches}
    return (
        matches,
        tuple(row for row in range(rows) if row not in matched_rows),
        tuple(col for col in range(cols) if col not in matched_cols),
    )


def iou_distance(tracks, detections):
    cost = np.ones((len(tracks), len(detections)), dtype=float)
    if not len(tracks) or not len(detections):
        return cost

    left = np.asarray([track.tlbr for track in tracks], dtype=float)[:, None, :]
    right = np.asarray([detection.tlbr for detection in detections], dtype=float)[None, :, :]
    overlap = np.maximum(np.minimum(left[..., 2:], right[..., 2:])
                         - np.maximum(left[..., :2], right[..., :2]), 0)
    intersection = overlap.prod(axis=2)
    left_area = np.maximum(left[..., 2:] - left[..., :2], 0).prod(axis=2)
    right_area = np.maximum(right[..., 2:] - right[..., :2], 0).prod(axis=2)
    union = left_area + right_area - intersection
    return cost - np.divide(intersection, union, out=np.zeros_like(cost), where=union > 0)


def fuse_score(iou_cost, detections):
    scores = np.asarray([detection.score for detection in detections])
    return 1 - (1 - iou_cost) * scores
