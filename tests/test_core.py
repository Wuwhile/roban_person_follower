from types import SimpleNamespace
import unittest

import numpy as np

from roban_person_follower.core import DepthCorrection, Follower, Settings, Track, torso_distance
from roban_person_follower.image import decode_image
from roban_person_follower.watchdog import FrameWatchdog


class DistanceTest(unittest.TestCase):
    box = (0, 0, 100, 100)

    def test_torso_median_uses_millimeters_and_ignores_background(self):
        depth = np.full((100, 100), 3000, dtype=np.uint16)
        depth[25:60, 30:70] = 1500
        depth[25:30, 30:70] = 0
        depth[30:34, 30:70] = 9000
        self.assertAlmostEqual(torso_distance(depth, self.box, 0.25), 1.5)

    def test_float_meters_and_nonfinite_depth(self):
        depth = np.full((100, 100), 1.5, dtype=np.float32)
        depth[25:30, :] = np.nan
        depth[30:35, :] = np.inf
        depth[35:40, :] = -1
        self.assertEqual(torso_distance(depth, self.box, 0.25), 1.5)

    def test_insufficient_valid_depth_and_offscreen_box(self):
        depth = np.zeros((100, 100), dtype=np.uint16)
        depth[25:28, 30:70] = 1500
        self.assertIsNone(torso_distance(depth, self.box, 0.25))
        self.assertIsNone(torso_distance(depth, (100, 0, 120, 20), 0.25))

    def test_inverse_depth_model_rejects_values_outside_its_measured_range(self):
        model = DepthCorrection(1.0, -.168, .70, 1.55)
        self.assertAlmostEqual(model.correct(1.173), 1.46088854, places=7)
        for value in (None, float("nan"), .699, 1.551):
            self.assertIsNone(model.correct(value))
        with self.assertRaises(ValueError):
            DepthCorrection(1.0, -.8, .70, 1.55)


class DecisionTest(unittest.TestCase):
    def setUp(self):
        self.follower = Follower(Settings())

    def decide(self, center=50, distance=1.5, track_id=1):
        track = Track(track_id, (center - 10, 0, center + 10, 40), distance)
        return self.follower.evaluate([track], 100)

    def test_commands_and_lock_release(self):
        self.assertEqual(self.decide().stop_reason, "idle")
        self.assertEqual(self.follower.command("start").stop_reason, "waiting_for_frame")
        self.assertIsNone(self.follower.command("start"))
        self.assertEqual(self.decide().action, "hold")
        self.assertEqual(self.follower.command("stop").stop_reason, "operator_stop")
        self.assertEqual(self.follower.track_id, 1)
        self.assertEqual(self.decide(distance=2).action, "stop")
        self.follower.command("start")
        self.assertEqual(self.decide(distance=2).action, "forward")
        self.assertEqual(self.follower.command("end").stop_reason, "operator_end")
        self.assertIsNone(self.follower.track_id)
        self.assertEqual(self.decide().stop_reason, "operator_end")
        self.assertIsNone(self.follower.track_id)
        with self.assertRaises(ValueError):
            self.follower.command("unknown")

    def test_direction_and_distance_boundaries(self):
        self.follower.command("start")
        self.assertEqual(self.decide(center=39, distance=2).action, "turn_left")
        self.assertEqual(self.decide(center=61, distance=2).action, "turn_right")
        self.assertEqual(self.decide(center=40, distance=float(np.float32(1.3))).action, "hold")
        self.assertEqual(self.decide(center=60, distance=float(np.float32(1.7))).action, "hold")
        self.assertEqual(self.decide(distance=1.701).action, "forward")

    def test_stop_reasons_take_priority_over_turns(self):
        self.follower.command("start")
        self.assertEqual(self.decide(center=20, distance=1.29).stop_reason, "too_close")
        for distance in (None, float("nan"), 0):
            self.assertEqual(self.decide(center=20, distance=distance).stop_reason, "invalid_depth")
        lost = self.follower.evaluate([], 100)
        self.assertEqual((lost.action, lost.stop_reason), ("stop", "target_lost"))
        self.assertIsNone(lost.target)

    def test_ambiguous_target_and_reacquisition(self):
        self.follower.command("start")
        tracks = [Track(i, (40, 0, 60, 40), 1.5) for i in (1, 2)]
        self.assertEqual(self.follower.evaluate(tracks, 100).stop_reason, "ambiguous_target")
        self.assertEqual(self.decide(track_id=2).action, "hold")
        self.assertEqual(self.follower.evaluate(tracks, 100).target.track_id, 2)
        self.assertEqual(self.follower.evaluate([], 100).stop_reason, "target_lost")
        self.assertEqual(self.decide(track_id=1).target.track_id, 1)


class ImageTest(unittest.TestCase):
    def test_rgb_to_bgr_and_row_padding(self):
        message = SimpleNamespace(encoding="rgb8", width=1, height=2, step=4,
                                  is_bigendian=False, data=bytes((1, 2, 3, 0, 4, 5, 6, 0)))
        np.testing.assert_array_equal(decode_image(message), [[[3, 2, 1]], [[6, 5, 4]]])

    def test_big_endian_aligned_depth(self):
        message = SimpleNamespace(encoding="16UC1", width=2, height=1, step=6,
                                  is_bigendian=True, data=b"\x05\xdc\x03\xe8\x00\x00")
        depth = decode_image(message)
        self.assertEqual(depth.dtype, np.uint16)
        np.testing.assert_array_equal(depth, [[1500, 1000]])
        message.step += 1
        with self.assertRaises(ValueError):
            decode_image(message)


class WatchdogTest(unittest.TestCase):
    def test_missing_frames_stop_once_and_recovery_rearms(self):
        watchdog = FrameWatchdog(timeout_s=0.5, started_at=10)
        self.assertFalse(watchdog.expired(10.5))
        self.assertTrue(watchdog.expired(10.55))
        self.assertFalse(watchdog.expired(11))
        watchdog.reset(11.1)
        self.assertFalse(watchdog.expired(11.4))
        self.assertTrue(watchdog.expired(11.7))


if __name__ == "__main__":
    unittest.main()
