from types import SimpleNamespace

from .bytetrack.byte_tracker import BYTETracker
from .core import Track, torso_distance


class PersonTracker:
    def __init__(self, min_depth_coverage: float, depth_correction=None):
        options = SimpleNamespace(
            track_thresh=0.4, match_thresh=0.8, track_buffer=30, mot20=False,
        )
        self.tracker = BYTETracker(options)
        self.min_depth_coverage = min_depth_coverage
        self.depth_correction = depth_correction

    def update(self, detections, depth):
        tracks = []
        for track in self.tracker.update(detections):
            box = tuple(float(value) for value in track.tlbr)
            distance = torso_distance(depth, box, self.min_depth_coverage)
            if self.depth_correction is not None:
                distance = self.depth_correction.correct(distance)
            tracks.append(Track(track.track_id, box, distance))
        return tracks
