from dataclasses import dataclass
from math import atan2, cos, degrees, isfinite, radians, sin
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class MotionSettings:
    camera_height_m: float = 0.56
    camera_forward_m: float = 0.05
    camera_pitch_deg: float = 10.0
    corridor_half_width_m: float = 0.23
    obstacle_stop_m: float = 0.95
    coverage_near_m: float = 0.65
    obstacle_min_height_m: float = 0.14
    obstacle_max_height_m: float = 0.54
    coverage_fraction: float = 0.20
    obstacle_pixels: int = 12
    head_yaw_sign: int = 1
    gait_yaw_sign: int = 1
    head_limit_deg: float = 18.0
    head_step_deg: float = 2.0
    head_settle_s: float = 0.45
    neutral_tolerance_deg: float = 2.0
    yaw_tolerance_deg: float = 8.0
    turn_delay_s: float = 0.85
    forward_step_m: float = 0.02
    turn_step_deg: float = 4.0
    perception_timeout_s: float = 0.5
    request_timeout_s: float = 1.5
    request_max_age_s: float = 0.30
    step_timeout_s: float = 4.5
    trial_timeout_s: float = 75.0
    camera_lateral_m: float = 0.0
    walk_head_pitch_deg: float = 0.0

    def __post_init__(self):
        if not all(type(value) in (int, float) and isfinite(value) for value in vars(self).values()):
            raise ValueError("Motion parameters must be finite numbers")
        if self.head_yaw_sign not in (-1, 1) or self.gait_yaw_sign not in (-1, 1):
            raise ValueError("Yaw signs must be -1 or 1")
        if not (0.15 <= self.camera_height_m <= 1.2
                and -0.20 <= self.camera_forward_m <= 0.30
                and abs(self.camera_lateral_m) <= 0.10
                and 0 <= self.walk_head_pitch_deg <= 20
                and -15 <= self.camera_pitch_deg <= 45
                and 0.12 <= self.corridor_half_width_m <= 0.45
                and 0.35 <= self.coverage_near_m < self.obstacle_stop_m < 1.3
                and 0.06 <= self.obstacle_min_height_m < self.obstacle_max_height_m <= 0.9
                and 0.1 <= self.coverage_fraction <= 1
                and type(self.obstacle_pixels) is int and self.obstacle_pixels >= 4):
            raise ValueError("Invalid obstacle or camera geometry")
        if not (0 < self.forward_step_m <= 0.05 and 0 < self.turn_step_deg <= 8
                and 0 < self.head_step_deg <= self.head_limit_deg <= 25
                and 0 < self.neutral_tolerance_deg <= 3
                and self.neutral_tolerance_deg < self.yaw_tolerance_deg <= 15
                and 0.2 <= self.head_settle_s < self.turn_delay_s
                and 0 < self.perception_timeout_s <= 0.5
                and 0 < self.request_max_age_s < self.request_timeout_s <= 2
                and 0 < self.step_timeout_s <= 6 and 5 <= self.trial_timeout_s <= 120):
            raise ValueError("Invalid motion limits or timeouts")


@dataclass(frozen=True)
class Corridor:
    visible: bool
    blocked: bool


class CorridorCheck:
    """Project neutral-camera depth into the corridor in front of the torso."""

    def __init__(self, settings, projection, width, height):
        self.settings = settings
        self.width, self.height = width, height
        self.fx, self.fy = projection[0], projection[5]
        self.cx, self.cy = projection[2], projection[6]
        if min(self.fx, self.fy) <= 0 or width < 1 or height < 1:
            raise ValueError("Uncalibrated RGB camera")
        self.sine = sin(radians(settings.camera_pitch_deg))
        self.cosine = cos(radians(settings.camera_pitch_deg))
        self.stride = 4
        rows, cols = np.mgrid[0:height:self.stride, 0:width:self.stride]
        self.ray_x = (cols - self.cx) / self.fx
        self.ray_y = (rows - self.cy) / self.fy
        self.coverage = self.coverage_regions(rows.shape)

    def coverage_regions(self, shape):
        cfg = self.settings
        regions = []
        for left, right in ((-cfg.corridor_half_width_m, 0), (0, cfg.corridor_half_width_m)):
            for low, high in ((cfg.obstacle_min_height_m, (cfg.obstacle_min_height_m+cfg.obstacle_max_height_m)/2),
                              ((cfg.obstacle_min_height_m+cfg.obstacle_max_height_m)/2, cfg.obstacle_max_height_m)):
                pixels = []
                for forward in (cfg.coverage_near_m, cfg.obstacle_stop_m):
                    relative_fwd = forward - cfg.camera_forward_m
                    for height in (low, high):
                        down = cfg.camera_height_m - height
                        optical_y = down*self.cosine - relative_fwd*self.sine
                        optical_z = relative_fwd*self.cosine + down*self.sine
                        if optical_z <= 0:
                            return None
                        for lateral in (left, right):
                            u = self.cx+self.fx*(lateral-cfg.camera_lateral_m)/optical_z
                            v = self.cy+self.fy*optical_y/optical_z
                            if not (self.stride <= u < self.width-self.stride
                                    and self.stride <= v < self.height-self.stride):
                                return None
                            pixels.append((u/self.stride, v/self.stride))
                mask = np.zeros(shape, dtype=np.uint8)
                hull = cv2.convexHull(np.round(pixels).astype(np.int32))
                cv2.fillConvexPoly(mask, hull, 1)
                regions.append(mask.astype(bool))
        return regions

    def measure(self, depth):
        if depth.shape != (self.height, self.width) or self.coverage is None:
            return Corridor(False, False)
        d = depth[::self.stride, ::self.stride].astype(np.float32)
        if depth.dtype == np.uint16:
            d *= .001
        valid = np.isfinite(d) & (d >= .175)
        safe_depth = np.where(valid, d, 0)
        down = self.ray_y * safe_depth
        cfg = self.settings
        lateral = cfg.camera_lateral_m + self.ray_x * safe_depth
        forward = cfg.camera_forward_m + self.cosine*safe_depth - self.sine*down
        height = cfg.camera_height_m - self.sine*safe_depth - self.cosine*down
        blocked = valid & (abs(lateral) <= cfg.corridor_half_width_m) & (forward > 0)
        blocked &= (forward <= cfg.obstacle_stop_m) & (height >= cfg.obstacle_min_height_m)
        blocked &= height <= cfg.obstacle_max_height_m
        # Each quadrant needs depth evidence to avoid treating a blind quadrant as open.
        visible = all(np.count_nonzero(valid & roi) >= cfg.coverage_fraction*np.count_nonzero(roi)
                      for roi in self.coverage)
        return Corridor(visible, np.count_nonzero(blocked) >= cfg.obstacle_pixels)


@dataclass(frozen=True)
class Perception:
    received: float
    enabled: bool
    distance_m: Optional[float]
    bearing_deg: Optional[float]
    corridor: Corridor


@dataclass(frozen=True)
class Intent:
    action: str
    value: float = 0.0
    reason: str = ""


def image_bearing(box, fx, cx):
    return degrees(atan2(cx-(box[0]+box[2])/2, fx))


class FollowMotion:
    def __init__(self, settings, near_m, far_m):
        self.settings = settings
        self.near_m = near_m
        self.far_m = far_m
        self.off_center_since = None
        self.off_center_side = None

    def decide(self, now, sample, head_deg):
        cfg = self.settings
        if sample is None or not 0 <= now-sample.received <= cfg.perception_timeout_s:
            return Intent("stop", reason="perception_timeout")
        if not sample.enabled:
            return Intent("stop", reason="perception_idle")
        if sample.bearing_deg is None or sample.distance_m is None:
            return Intent("stop", reason="target_or_depth_lost")
        if sample.distance_m < self.near_m - 1e-6:
            return Intent("stop", reason="too_close")
        neutral = abs(head_deg) <= cfg.neutral_tolerance_deg
        if neutral and sample.corridor.blocked:
            return Intent("stop", reason="obstacle")

        bearing = head_deg + sample.bearing_deg
        side = 1 if bearing > cfg.yaw_tolerance_deg else -1 if bearing < -cfg.yaw_tolerance_deg else 0
        if side != self.off_center_side:
            self.off_center_since = now
            self.off_center_side = side
        if side and now-self.off_center_since < cfg.turn_delay_s:
            desired = max(-cfg.head_limit_deg, min(cfg.head_limit_deg, bearing))
            delta = max(-cfg.head_step_deg, min(cfg.head_step_deg, desired-head_deg))
            if abs(delta) >= 1:
                return Intent("head", head_deg+delta)
            return Intent("hold", reason="head_tracking")
        if abs(head_deg) >= 1:
            delta = max(-cfg.head_step_deg, min(cfg.head_step_deg, -head_deg))
            return Intent("head", head_deg+delta, "recenter_for_gait")
        if side or sample.distance_m > self.far_m + 1e-6:
            if not sample.corridor.visible:
                return Intent("stop", reason="forward_coverage_lost")
            if side:
                return Intent("turn", side*cfg.turn_step_deg)
            return Intent("forward", cfg.forward_step_m)
        return Intent("hold", reason="distance_band")
