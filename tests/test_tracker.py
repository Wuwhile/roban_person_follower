import os
from pathlib import Path
import unittest

import cv2
import numpy as np

from roban_person_follower.detector import PersonDetector
from roban_person_follower.model import MODEL_SHA256, sha256
from roban_person_follower.tracker import PersonTracker


class TrackerTest(unittest.TestCase):
    def test_high_then_low_confidence_preserves_id_and_empty_frame_clears_target(self):
        tracker = PersonTracker(0.25)
        depth = np.full((100, 100), 1500, dtype=np.uint16)
        high_score = np.array([[20, 15, 80, 95, 0.9]], dtype=np.float32)
        low_score = np.array([[21, 15, 81, 95, 0.25]], dtype=np.float32)
        first = tracker.update(high_score, depth)[0]
        continued = tracker.update(low_score, depth)[0]
        self.assertEqual(first.track_id, continued.track_id)
        self.assertAlmostEqual(continued.distance_m, 1.5)
        self.assertEqual(tracker.update(np.empty((0, 5), dtype=np.float32), depth), [])


@unittest.skipUnless(os.environ.get("ROBAN_TEST_MODEL") and os.environ.get("ROBAN_TEST_IMAGE"),
                     "Set ROBAN_TEST_MODEL and ROBAN_TEST_IMAGE for the real CPU model smoke test")
class RealModelTest(unittest.TestCase):
    def test_official_yolox_on_cpu_with_person_photo_and_synthetic_depth(self):
        model = Path(os.environ["ROBAN_TEST_MODEL"])
        self.assertEqual(sha256(model), MODEL_SHA256)
        image = cv2.imread(os.environ["ROBAN_TEST_IMAGE"], cv2.IMREAD_COLOR)
        self.assertIsNotNone(image, "Cannot open ROBAN_TEST_IMAGE")
        detector = PersonDetector(model, threads=2)
        self.assertEqual(detector.session.get_providers(), ["CPUExecutionProvider"])
        detections = detector.detect(image)
        self.assertGreaterEqual(len(detections), 1)
        depth = np.full(image.shape[:2], 1500, dtype=np.uint16)
        tracks = PersonTracker(0.25).update(detections, depth)
        self.assertGreaterEqual(len(tracks), 1)
        for track in tracks:
            self.assertAlmostEqual(track.distance_m, 1.5)


if __name__ == "__main__":
    unittest.main()
