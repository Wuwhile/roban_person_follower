import csv
from datetime import datetime, timezone
import os
from pathlib import Path
from time import monotonic


FIELDS = (
    "event", "publish_ros_s", "publish_monotonic_s", "rgb_stamp_s",
    "depth_stamp_s", "sync_delta_ms", "mode", "enabled", "track_id",
    "distance_m", "decision", "stop_reason", "latency_ms",
)


class RunLog:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.path = directory / f"run-{stamp}-{os.getpid()}.csv"
        self.file = self.path.open("x", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(FIELDS)
        self.file.flush()

    def record(self, event, publish_s, mode, enabled, decision, distance, pair=None):
        target = decision.target
        rgb_s = pair.rgb.header.stamp.to_sec() if pair else None
        depth_s = pair.depth.header.stamp.to_sec() if pair else None
        sync_ms = abs(rgb_s - depth_s) * 1e3 if pair else None
        latency_ms = (publish_s - rgb_s) * 1e3 if pair else None
        self.writer.writerow((
            event, publish_s, monotonic(), rgb_s, depth_s, sync_ms, mode, int(enabled),
            target.track_id if target else None, distance, decision.action, decision.stop_reason,
            max(latency_ms, 0) if latency_ms is not None else None,
        ))
        self.file.flush()

    def close(self):
        self.file.close()
