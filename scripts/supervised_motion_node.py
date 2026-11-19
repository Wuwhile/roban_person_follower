#!/usr/bin/env python3
import argparse
from collections import deque
import csv
from dataclasses import fields, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import sys
from threading import RLock
from time import monotonic

import message_filters
import rospy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from roban_person_follower.bodyhub import BodyHub
from roban_person_follower.core import Settings, usable_distance
from roban_person_follower.image import decode_image
from roban_person_follower.model import ros_data_dir
from roban_person_follower.motion_policy import CorridorCheck, FollowMotion, MotionSettings, Perception, image_bearing
from roban_person_follower.motion_session import MotionSession
from roban_person_follower.msg import Status, Target


class MotionNode:
    def __init__(self):
        defaults = MotionSettings()
        self.cfg = MotionSettings(**{field.name: rospy.get_param("~"+field.name, getattr(defaults, field.name))
                                     for field in fields(defaults)})
        self.near = float(rospy.get_param("~near_m", Settings().near_m))
        self.far = float(rospy.get_param("~far_m", Settings().far_m))
        if not 0 < self.near < self.far:
            raise ValueError("Invalid follow band")
        self.enable_motion = rospy.get_param("~enable_motion", False) is True
        self.calibrated = rospy.get_param("~field_calibrated", False) is True
        self.lock = RLock()
        self.queue = deque()
        self.revoked = ""
        self.sample = None
        self.follower_enabled = False
        self.camera = None
        self.camera_key = None
        self.state_guard = None
        state_dir = Path(rospy.get_param("~state_dir", "") or ros_data_dir()).expanduser()
        self.latch_path = state_dir / "estop.latched"
        self.estop = self.latch_path.exists()
        self.status = {}
        self.last_published = 0.0
        self.pub = rospy.Publisher("motion_status", String, queue_size=2, latch=True)
        log_dir = Path(rospy.get_param("~log_dir", "") or ros_data_dir()/"logs").expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.log = (log_dir/f"motion-{stamp}.csv").open("x", newline="", encoding="utf-8")
        self.writer = csv.writer(self.log)
        self.writer.writerow(("ros_time_s", "monotonic_s", "phase", "action", "reason", "steps"))
        self.log.flush()
        self.device = self.session = None
        if self.enable_motion and self.calibrated:
            import fcntl
            state_dir.mkdir(parents=True, exist_ok=True)
            self.state_guard = (state_dir/"motion.lock").open("a")
            fcntl.flock(self.state_guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.device = BodyHub(rospy, state_dir/"motion-gait-journal.json")
            self.session = MotionSession(self.device, self.cfg, self.near, self.far, self.inputs, self.notify, self.lock)

        self.preview = FollowMotion(self.cfg, self.near, self.far)
        self.status_sub = rospy.Subscriber("status", Status, self.receive_status, queue_size=5)
        self.command_sub = rospy.Subscriber("command", String, self.receive_follower_command, queue_size=5)
        self.motion_sub = rospy.Subscriber("motion_command", String, self.receive_motion_command, queue_size=5)
        self.info_sub = rospy.Subscriber("depth_registered/camera_info", CameraInfo, self.receive_info, queue_size=1)
        self.target_sub = message_filters.Subscriber("target", Target, queue_size=3)
        self.depth_sub = message_filters.Subscriber("depth_registered/image_rect", Image, queue_size=3, buff_size=2**22)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.target_sub, self.depth_sub], queue_size=6, slop=Settings().sync_slop_s,
        )
        self.sync.registerCallback(self.receive_pair)
        self.notify("estop" if self.estop else "standby", "none", "startup", 0)

    def receive_status(self, msg):
        with self.lock:
            self.follower_enabled = msg.enabled and msg.stop_reason not in (
                "frame_timeout", "inference_error", "invalid_frame",
            )

    def receive_follower_command(self, msg):
        value = msg.data.strip().lower()
        if value in ("stop", "end"):
            self.receive_motion_command(String(data="stop"))

    def receive_motion_command(self, msg):
        value = msg.data.strip().lower()
        if value not in ("arm", "arm_once", "stop", "estop", "reset_estop"):
            rospy.logwarn_throttle(5, "Expected arm, arm_once, stop, estop or reset_estop")
            return
        with self.lock:
            if value in ("stop", "estop"):
                self.queue.clear()
                if value == "estop":
                    self.estop = True
                    try:
                        self.latch_path.parent.mkdir(parents=True, exist_ok=True)
                        self.latch_path.touch()
                    except OSError as error:
                        rospy.logerr("Cannot persist emergency-stop latch: %s", error)
                self.revoked = "estop" if self.estop else "operator_stop"
            self.queue.append(value)

    def receive_info(self, msg):
        key = (msg.header.frame_id, tuple(msg.P), msg.width, msg.height)
        if self.camera_key == key:
            return
        try:
            camera = CorridorCheck(self.cfg, msg.P, msg.width, msg.height)
        except (ValueError, OverflowError):
            camera = None
        with self.lock:
            self.camera_key, self.camera = key, camera

    def receive_pair(self, target, depth):
        stamp = rospy.Time.now()
        age = max((stamp-target.header.stamp).to_sec(), (stamp-depth.header.stamp).to_sec())
        if min(target.header.stamp.to_sec(), depth.header.stamp.to_sec()) <= 0:
            return
        if not -Settings().sync_slop_s <= age <= self.cfg.perception_timeout_s:
            return
        with self.lock:
            camera, key = self.camera, self.camera_key
        if camera is None or key[0] != target.header.frame_id or key[0] != depth.header.frame_id:
            return
        try:
            corridor = camera.measure(decode_image(depth))
        except (TypeError, ValueError):
            return
        bearing = image_bearing(target.bbox_xyxy, camera.fx, camera.cx) if target.valid else None
        if bearing is not None and not -90 < bearing < 90:
            bearing = None
        distance = target.distance_m if target.distance_valid and usable_distance(target.distance_m) else None
        sample = Perception(monotonic()-max(0, age), True, distance, bearing, corridor)
        with self.lock:
            self.sample = sample

    def inputs(self):
        with self.lock:
            sample = replace(self.sample, enabled=self.follower_enabled) if self.sample else None
            return sample, self.revoked or ("estop" if self.estop else "")

    def notify(self, phase, action, reason, steps):
        next_status = dict(phase=phase, action=action, reason=reason, steps=steps)
        if next_status != self.status:
            self.status = next_status
            self.writer.writerow((rospy.get_time(), monotonic(), phase, action, reason, steps))
            self.log.flush()
            self.publish_status()

    def publish_status(self):
        self.pub.publish(String(data=json.dumps(self.status)))
        self.last_published = monotonic()

    def command(self, value):
        if value == "estop":
            if self.session:
                self.session.stop("estop")
            else:
                self.notify("estop", "none", "estop", 0)
        elif value == "stop":
            if self.session and (self.session.active or self.session.fault):
                self.session.stop("estop" if self.estop else "operator_stop")
            elif not self.estop:
                self.notify("standby", "none", "operator_stop", 0)
        elif value == "reset_estop":
            if self.session and (self.session.active or self.session.fault
                                 or not self.hub_idle()):
                return
            with self.lock:
                if "estop" in self.queue:
                    return
                self.latch_path.unlink(missing_ok=True)
                self.estop = False
                self.revoked = ""
            self.notify("standby", "none", "manual_reset", 0)
        elif not self.session:
            self.notify("standby", "none", "calibration_required" if not self.calibrated
                        else "hardware_disabled", 0)
        else:
            with self.lock:
                if self.estop or any(item in ("stop", "estop") for item in self.queue):
                    return
                self.revoked = ""
            self.session.arm(value == "arm_once")

    def hub_idle(self):
        try:
            return (self.device.owner(), self.device.state()) == (0, "preReady")
        except Exception as error:
            rospy.logerr("Cannot confirm BodyHub idle for reset: %s", error)
            return False

    def run(self):
        try:
            while not rospy.is_shutdown():
                with self.lock:
                    value = self.queue.popleft() if self.queue else None
                if value:
                    self.command(value)
                if self.session:
                    self.session.tick()
                elif not self.estop:
                    sample, revoked = self.inputs()
                    if not revoked and sample and sample.enabled:
                        intent = self.preview.decide(monotonic(), sample, 0.0)
                        self.notify("shadow", intent.action, intent.reason, 0)
                if monotonic()-self.last_published > .25:
                    self.publish_status()
                rospy.rostime.wallsleep(.04)
        finally:
            try:
                if self.session and (self.session.active or self.session.fault):
                    self.session.stop("shutdown")
            finally:
                self.log.close()
                if self.state_guard:
                    self.state_guard.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recover-sources", action="store_true")
    parser.add_argument("--state-dir", type=Path, default=ros_data_dir())
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])
    rospy.init_node("motion_recovery" if args.recover_sources else "motion")
    socket.setdefaulttimeout(2)
    if args.recover_sources:
        import fcntl
        with (args.state_dir/"motion.lock").open("a") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            BodyHub(rospy, args.state_dir/"motion-gait-journal.json").recover_idle()
        rospy.loginfo("BodyHub idle and prior publisher isolation cleared")
    else:
        MotionNode().run()


if __name__ == "__main__":
    main()
