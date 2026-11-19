#!/usr/bin/env python3
from dataclasses import dataclass, fields
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic

import cv2
import message_filters
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String

from roban_person_follower.core import Decision, Follower, Settings, usable_distance
from roban_person_follower.detector import PersonDetector
from roban_person_follower.image import decode_image
from roban_person_follower.model import default_model_path, ros_data_dir
from roban_person_follower.msg import Status, Target
from roban_person_follower.run_log import RunLog
from roban_person_follower.tracker import PersonTracker
from roban_person_follower.watchdog import FrameWatchdog


@dataclass(frozen=True)
class FramePair:
    rgb: Image
    depth: Image
    received_at: float

    def fresh(self, settings):
        now = rospy.Time.now()
        if now.is_zero() or monotonic() - self.received_at > settings.frame_timeout_s:
            return False
        return all(
            not message.header.stamp.is_zero()
            and -settings.sync_slop_s <= (now - message.header.stamp).to_sec()
            <= settings.frame_timeout_s
            for message in (self.rgb, self.depth)
        )

    def decode(self):
        if not self.rgb.header.frame_id or self.rgb.header.frame_id != self.depth.header.frame_id:
            raise ValueError("Rectified RGB and registered depth have different optical frames")
        rgb, depth = decode_image(self.rgb), decode_image(self.depth)
        if rgb.ndim != 3 or depth.ndim != 2 or rgb.shape[:2] != depth.shape:
            raise ValueError("Rectified RGB and registered depth must have identical dimensions")
        return rgb, depth


class FollowerNode:
    def __init__(self):
        defaults = Settings()
        self.settings = Settings(**{
            field.name: rospy.get_param("~" + field.name, getattr(defaults, field.name))
            for field in fields(defaults)
        })
        self.mode = rospy.get_param("~mode", "observe")
        if self.mode not in ("observe", "live"):
            raise ValueError("mode must be observe or live")
        model = Path(rospy.get_param("~model_path", "") or default_model_path()).expanduser()
        log_dir = Path(rospy.get_param("~log_dir", "") or ros_data_dir() / "logs").expanduser()

        cv2.setNumThreads(1)
        self.detector = PersonDetector(model, self.settings.inference_threads)
        self.tracker = PersonTracker(self.settings.min_depth_coverage)
        self.follower = Follower(self.settings)
        self.target_pub = rospy.Publisher("target", Target, queue_size=1, latch=True)
        self.status_pub = rospy.Publisher("status", Status, queue_size=1, latch=True)
        self.debug_pub = rospy.Publisher("debug_image", Image, queue_size=1, latch=True)

        self.lock = Lock()
        self.ready = Event()
        self.closed = Event()
        self.pending = None
        self.command_generation = 0
        self.last_image = np.zeros((240, 480, 3), dtype=np.uint8)
        self.last_header = Header()
        self.watchdog = FrameWatchdog(self.settings.frame_timeout_s, monotonic())
        self.worker_error = None
        self.worker = Thread(target=self.process_frames, name="rgbd-inference")
        self.log = RunLog(log_dir)
        try:
            with self.lock:
                self.publish("startup", Decision("stop", "idle"))
                self.rgb_sub = message_filters.Subscriber(
                    "rectified_color", Image, queue_size=1, buff_size=2 ** 22,
                )
                self.depth_sub = message_filters.Subscriber(
                    "depth_registered/image_rect", Image, queue_size=1, buff_size=2 ** 22,
                )
                self.sync = message_filters.ApproximateTimeSynchronizer(
                    [self.rgb_sub, self.depth_sub], queue_size=5, slop=self.settings.sync_slop_s,
                )
                self.sync.registerCallback(self.receive_pair)
                self.command_sub = rospy.Subscriber("command", String, self.receive_command, queue_size=10)
        except Exception:
            self.log.close()
            raise
        rospy.loginfo("Follower mode=%s with decision output only. Run log: %s", self.mode, self.log.path)

    def receive_pair(self, rgb, depth):
        pair = FramePair(rgb, depth, monotonic())
        with self.lock:
            if self.closed.is_set() or not pair.fresh(self.settings):
                return
            if self.pending is None or rgb.header.stamp > self.pending.rgb.header.stamp:
                self.pending = pair
                self.ready.set()

    def receive_command(self, message):
        with self.lock:
            if self.closed.is_set():
                return
            try:
                decision = self.follower.command(message.data.strip().lower())
            except ValueError:
                rospy.logwarn_throttle(5, "Unknown follower command. Use start, stop or end.")
                return
            if decision is None:
                return
            self.command_generation += 1
            self.pending = None
            self.watchdog.reset(monotonic())
            self.publish("command", decision)

    def process_frames(self):
        try:
            while not self.closed.is_set():
                self.ready.wait()
                with self.lock:
                    pair, self.pending = self.pending, None
                    generation = self.command_generation
                    self.ready.clear()
                if pair is None or not pair.fresh(self.settings):
                    continue
                try:
                    bgr, depth = pair.decode()
                except ValueError as error:
                    rospy.logwarn_throttle(5, str(error))
                    self.report_frame_error(pair, generation, "invalid_frame")
                    continue
                try:
                    tracks = self.tracker.update(self.detector.detect(bgr), depth)
                except Exception as error:
                    rospy.logerr_throttle(5, f"Person inference failed: {error}")
                    self.report_frame_error(pair, generation, "inference_error")
                    continue

                with self.lock:
                    if not self.can_publish(pair, generation):
                        continue
                    self.last_image, self.last_header = bgr, pair.rgb.header
                    decision = self.follower.evaluate(tracks, bgr.shape[1])
                    self.publish("frame", decision, pair, tracks)
        except Exception as error:
            self.worker_error = error
        finally:
            self.closed.set()

    def can_publish(self, pair, generation):
        if self.closed.is_set() or generation != self.command_generation:
            return False
        if not pair.fresh(self.settings):
            return False
        self.watchdog.reset(pair.received_at)
        return True

    def report_frame_error(self, pair, generation, reason):
        with self.lock:
            if self.can_publish(pair, generation):
                self.publish("frame_error", self.follower.unavailable(reason), pair)

    def publish(self, event, decision, pair=None, tracks=()):
        target = decision.target
        distance = target.distance_m if target and usable_distance(target.distance_m) else None
        debug = self.annotate(decision, tracks)
        now = rospy.Time.now()
        header = pair.rgb.header if pair else Header()
        latency = max((now - header.stamp).to_sec() * 1e3, 0) if pair else float("nan")
        self.status_pub.publish(Status(
            header=Header(stamp=now),
            frame_stamp=header.stamp,
            mode=self.mode,
            enabled=self.follower.enabled,
            decision=decision.action,
            stop_reason=decision.stop_reason,
            latency_ms=latency,
        ))
        self.target_pub.publish(Target(
            header=header,
            valid=target is not None,
            track_id=target.track_id if target else -1,
            bbox_xyxy=target.box if target else [0.0] * 4,
            distance_valid=distance is not None,
            distance_m=distance if distance is not None else float("nan"),
        ))
        self.debug_pub.publish(debug)
        self.log.record(event, now.to_sec(), self.mode, self.follower.enabled, decision, distance, pair)

    def annotate(self, decision, tracks):
        image = self.last_image.copy()
        height, width = image.shape[:2]
        for track in tracks:
            selected = decision.target is not None and track.track_id == decision.target.track_id
            color = (60, 230, 100) if selected else (40, 200, 255)
            x1, y1, x2, y2 = (int(value) for value in track.box)
            left, right = max(0, x1), min(width - 1, x2)
            top, bottom = max(0, y1), min(height - 1, y2)
            if right <= left or bottom <= top:
                continue
            distance = f"{track.distance_m:.2f} m" if usable_distance(track.distance_m) else "no depth"
            cv2.rectangle(image, (left, top), (right, bottom), color, 2)
            cv2.putText(image, f"#{track.track_id} {distance}", (left, max(44, top - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        label = f"{self.mode.upper()} | {decision.action.upper()}"
        if decision.stop_reason:
            label += f" | {decision.stop_reason}"
        cv2.rectangle(image, (0, 0), (width, 27), (24, 24, 24), cv2.FILLED)
        cv2.putText(image, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (235, 235, 235), 1, cv2.LINE_AA)
        return Image(
            header=self.last_header, height=height, width=width, encoding="bgr8",
            is_bigendian=0, step=width * 3, data=image.tobytes(),
        )

    def run(self):
        self.worker.start()
        try:
            while not rospy.is_shutdown() and not self.closed.wait(0.05):
                with self.lock:
                    if self.watchdog.expired(monotonic()):
                        self.publish("watchdog", self.follower.unavailable("frame_timeout"))
        finally:
            with self.lock:
                self.closed.set()
                self.ready.set()
            self.worker.join()
            with self.lock:
                self.log.close()
        if self.worker_error is not None:
            raise RuntimeError("Follower worker exited unexpectedly") from self.worker_error


def main():
    rospy.init_node("follower")
    try:
        FollowerNode().run()
    except Exception as error:
        rospy.logfatal("Follower failed: %s", error)
        raise


if __name__ == "__main__":
    main()
