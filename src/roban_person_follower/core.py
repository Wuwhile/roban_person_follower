from dataclasses import dataclass
from math import ceil, floor, isfinite
from typing import Optional, Sequence, Tuple

import numpy as np


Box = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Settings:
    near_m: float = 1.3
    far_m: float = 1.7
    center_fraction: float = 0.20
    sync_slop_s: float = 0.025
    frame_timeout_s: float = 0.5
    min_depth_coverage: float = 0.25
    inference_threads: int = 2

    def __post_init__(self):
        values = (
            self.near_m, self.far_m, self.center_fraction, self.sync_slop_s,
            self.frame_timeout_s, self.min_depth_coverage,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("Follower settings must be finite")
        if not 0 < self.near_m < self.far_m:
            raise ValueError("Expected 0 < near_m < far_m")
        if not 0 < self.center_fraction < 1:
            raise ValueError("center_fraction must be between 0 and 1")
        if not 0 < self.sync_slop_s < self.frame_timeout_s:
            raise ValueError("Expected 0 < sync_slop_s < frame_timeout_s")
        if not 0 < self.min_depth_coverage <= 1:
            raise ValueError("min_depth_coverage must be in (0, 1]")
        if type(self.inference_threads) is not int or self.inference_threads < 1:
            raise ValueError("inference_threads must be a positive integer")


@dataclass(frozen=True)
class Track:
    track_id: int
    box: Box
    distance_m: Optional[float]


@dataclass(frozen=True)
class Decision:
    action: str
    stop_reason: str = ""
    target: Optional[Track] = None


def usable_distance(distance_m: Optional[float]) -> bool:
    return distance_m is not None and isfinite(distance_m) and distance_m > 0


@dataclass(frozen=True)
class DepthCorrection:
    inverse_scale: float
    inverse_offset_per_m: float
    raw_min_m: float
    raw_max_m: float

    def __post_init__(self):
        if not all(type(value) in (int, float) and isfinite(value) for value in vars(self).values()):
            raise ValueError("Depth correction parameters must be finite numbers")
        if not 0 < self.raw_min_m < self.raw_max_m or self.inverse_scale <= 0:
            raise ValueError("Invalid depth correction range or scale")
        if self.inverse_scale / self.raw_max_m + self.inverse_offset_per_m <= 0:
            raise ValueError("Depth correction must remain positive throughout its range")

    def correct(self, raw_m: Optional[float]) -> Optional[float]:
        if not usable_distance(raw_m) or not self.raw_min_m <= raw_m <= self.raw_max_m:
            return None
        distance = 1.0 / (self.inverse_scale / raw_m + self.inverse_offset_per_m)
        return distance if usable_distance(distance) else None


def torso_distance(depth: np.ndarray, box: Box, min_coverage: float) -> Optional[float]:
    height, width = depth.shape
    left, top, right, bottom = box
    if not all(isfinite(value) for value in box) or right <= left or bottom <= top:
        return None

    box_width = right - left
    box_height = bottom - top
    x0 = max(0, floor(left + 0.30 * box_width))
    x1 = min(width, ceil(left + 0.70 * box_width))
    y0 = max(0, floor(top + 0.25 * box_height))
    y1 = min(height, ceil(top + 0.60 * box_height))
    if x1 <= x0 or y1 <= y0:
        return None

    torso = depth[y0:y1, x0:x1]
    valid = torso[np.isfinite(torso) & (torso > 0)]
    if valid.size < max(16, min_coverage * torso.size):
        return None
    scale = 0.001 if torso.dtype.kind == "u" and torso.dtype.itemsize == 2 else 1.0
    return float(np.median(valid)) * scale


class Follower:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = False
        self.track_id: Optional[int] = None
        self.idle_reason = "idle"

    def command(self, value: str) -> Optional[Decision]:
        if value == "start":
            if self.enabled:
                return None
            self.enabled = True
            return self.unavailable("waiting_for_frame")
        if value == "stop":
            self.enabled = False
            self.idle_reason = "operator_stop"
            return self.unavailable(self.idle_reason)
        if value == "end":
            self.enabled = False
            self.track_id = None
            self.idle_reason = "operator_end"
            return self.unavailable(self.idle_reason)
        raise ValueError("Expected start, stop or end")

    def unavailable(self, reason: str) -> Decision:
        return Decision("stop", reason)

    def evaluate(self, tracks: Sequence[Track], image_width: int) -> Decision:
        target = next((track for track in tracks if track.track_id == self.track_id), None)
        if target is None and len(tracks) == 1:
            target = tracks[0]
        if not self.enabled:
            return Decision("stop", self.idle_reason, target)
        if target is None:
            reason = "ambiguous_target" if tracks else "target_lost"
            return self.unavailable(reason)

        self.track_id = target.track_id
        distance = target.distance_m
        if not usable_distance(distance):
            return Decision("stop", "invalid_depth", target)
        if distance < self.settings.near_m - 1e-6:
            return Decision("stop", "too_close", target)

        left, _, right, _ = target.box
        offset = (left + right) / (2 * image_width) - 0.5
        tolerance = self.settings.center_fraction / 2
        if offset < -tolerance - 1e-9:
            return Decision("turn_left", target=target)
        if offset > tolerance + 1e-9:
            return Decision("turn_right", target=target)
        action = "forward" if distance > self.settings.far_m + 1e-6 else "hold"
        return Decision(action, target=target)
