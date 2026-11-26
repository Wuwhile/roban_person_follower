import unittest
from dataclasses import replace
from threading import RLock

import numpy as np

from roban_person_follower.bodyhub import OWNER, BodyHub
from roban_person_follower.motion_policy import Corridor, CorridorCheck, FollowMotion, MotionSettings, Perception
from roban_person_follower.motion_session import MotionSession


class CorridorTest(unittest.TestCase):
    def setUp(self):
        cfg = MotionSettings(camera_pitch_deg=22)
        self.check = CorridorCheck(cfg, [600,0,320,0,0,600,240,0,0,0,1,0], 640, 480)

    def test_body_front_stop_with_clear_distinct_from_invalid(self):
        far = np.full((480,640), 2100, dtype=np.uint16)
        self.assertEqual(self.check.measure(far), Corridor(True, False))
        blocked = far.copy()
        blocked[170:290, 260:380] = 740
        self.assertEqual(self.check.measure(blocked), Corridor(True, True))
        far[:, :320] = 0
        self.assertEqual(self.check.measure(far), Corridor(False, False))

    def test_corridor_geometry_outside_image_disables_forward(self):
        cfg = MotionSettings(camera_pitch_deg=-10)
        check = CorridorCheck(cfg, [600,0,320,0,0,600,240,0,0,0,1,0], 640, 480)
        self.assertEqual(check.measure(np.full((480,640), 2500, dtype=np.uint16)),
                         Corridor(False, False))

    def test_lateral_camera_offset_places_obstacle_relative_to_torso(self):
        frame = np.full((480,640), 2100, dtype=np.uint16)
        frame[196:248, 488:512] = 740
        projection = [600,0,320,0,0,600,240,0,0,0,1,0]
        cfg = MotionSettings(camera_pitch_deg=22, camera_lateral_m=.06)
        check = CorridorCheck(cfg, projection, 640, 480)
        self.assertEqual(self.check.measure(frame), Corridor(True, True))
        self.assertEqual(check.measure(frame), Corridor(True, False))


class PolicyTest(unittest.TestCase):
    def setUp(self):
        self.policy = FollowMotion(MotionSettings(), 1.3, 1.7)

    def sample(self, now, bearing=0, distance=2, visible=True, blocked=False):
        return Perception(now, True, distance, bearing, Corridor(visible, blocked))

    def test_turn_head_first_then_recenter_and_turn(self):
        self.assertEqual(self.policy.decide(10, self.sample(10,14), 0).action, "head")
        recent = self.policy.decide(10.4, self.sample(10.4,9), 5)
        self.assertEqual(recent.action, "head")
        center = self.policy.decide(11, self.sample(11,9), 5)
        self.assertEqual((center.action, center.value), ("head", 3))
        turn = self.policy.decide(11.6, self.sample(11.6,14), 0)
        self.assertEqual((turn.action, turn.value), ("turn", 4))

    def test_visibility_and_range(self):
        self.assertEqual(self.policy.decide(10, self.sample(10, distance=1.7), 0).action, "hold")
        self.assertEqual(self.policy.decide(10, self.sample(10), 0).action, "forward")
        self.assertEqual(self.policy.decide(10, self.sample(10, visible=False), 0).reason,
                         "forward_coverage_lost")
        self.assertEqual(self.policy.decide(10, self.sample(10, blocked=True), 0).reason, "obstacle")
        self.assertEqual(self.policy.decide(10, self.sample(10, distance=1.29), 0).reason, "too_close")
        self.assertEqual(self.policy.decide(11, self.sample(10), 0).reason, "perception_timeout")


class MockDevice:
    def __init__(self, clock):
        self.clock = clock
        self.trace = []
        self.done = False
        self.request = True
        self.before_gait = lambda: None
        self.stop_error = False
        self.current_owner = OWNER

    def claim(self, permit):
        if not permit():
            raise RuntimeError("permit")
        self.trace.append("claim")

    def check_exclusive(self):
        pass

    def owner(self):
        return self.current_owner

    def head(self, angle, authorize, lock):
        with lock:
            if authorize():
                self.trace.append(("head", angle))
                return True
        return False

    def start_walking(self, tolerance, permit):
        if not permit():
            raise RuntimeError("permit")
        self.trace.append("walking_ready")
        return self.clock()

    def gait(self, forward, turn, max_age, authorize, lock):
        self.before_gait()
        with lock:
            if self.request and authorize():
                self.trace.append(("gait", forward, turn))
                return self.clock()
        return None

    def gait_finished(self, start):
        return self.done

    def stand(self):
        self.trace.append("stand")

    def close(self):
        self.trace.append("stop")
        if self.stop_error:
            raise RuntimeError("unconfirmed")


class SessionTest(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.revoked = ""
        self.cfg = MotionSettings()
        self.sample = Perception(self.now, True, 2.0, 0.0, Corridor(True, False))
        self.device = MockDevice(lambda: self.now)
        self.history = []
        self.session = MotionSession(self.device, self.cfg, 1.3, 1.7,
                                     lambda: (self.sample,self.revoked),
                                     lambda *data: self.history.append(data), RLock(), lambda: self.now)

    def tick(self, increment, **changes):
        self.now += increment
        self.sample = replace(self.sample, received=self.now, **changes)
        self.session.tick()

    def entered_walking(self, once=False):
        self.session.arm(once)
        self.tick(.5)
        self.assertIn("walking_ready", self.device.trace)

    def gears(self):
        return [event for event in self.device.trace if isinstance(event, tuple) and event[0]=="gait"]

    def test_one_step_blocks_additional_request_then_releases(self):
        self.entered_walking(once=True)
        self.tick(.5)
        self.tick(.3)
        self.assertEqual(self.gears(), [("gait", .02, 0)])
        self.device.done = True
        self.tick(.2)
        self.assertEqual(self.history[-1], ("standby","none","single_step_complete",1))

    def test_timeout_sends_no_gait(self):
        self.device.request = False
        self.entered_walking()
        self.tick(.5)
        self.tick(1.6)
        self.assertEqual(self.gears(), [])
        self.assertEqual(self.history[-1][2], "gait_request_timeout")

    def test_stale_or_obstacle_stops_inflight(self):
        self.entered_walking()
        self.tick(.5)
        self.tick(.1, corridor=Corridor(False, False))
        self.assertEqual(self.history[-1][2], "forward_coverage_lost")
        self.assertEqual(len(self.gears()), 1)

    def test_revoke_while_waiting_for_hardware_prevents_publication(self):
        self.entered_walking()
        self.device.before_gait = lambda: setattr(self, "revoked", "estop")
        self.tick(.5)
        self.assertEqual(self.gears(), [])
        self.tick(.05)
        self.assertEqual(self.history[-1][0], "estop")

    def test_bodyhub_stop_unconfirmed_blocks_rearming(self):
        self.entered_walking()
        self.device.stop_error = True
        self.tick(.5, bearing_deg=None)
        self.assertEqual(self.history[-1][0], "stop_unconfirmed")
        before = self.device.trace.copy()
        self.session.arm(False)
        self.assertEqual(before, self.device.trace)

    def test_request_consumption_and_failed_cycle(self):
        self.entered_walking()
        self.tick(.5)
        self.tick(4.6)
        self.assertEqual(self.history[-1][2], "walking_feedback_timeout")
        self.assertEqual(len(self.gears()), 1)

    def test_transport_feedback_must_contain_ordered_motion_edges(self):
        hub = object.__new__(BodyHub)
        hub.walking = [(9.9,True), (10.01,False), (10.02,True)]
        self.assertFalse(hub.gait_finished(10))
        hub.walking.append((10.04,False))
        self.assertTrue(hub.gait_finished(10))


if __name__ == "__main__":
    unittest.main()
