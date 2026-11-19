#!/usr/bin/env python3
import argparse
from collections import Counter
import csv
from math import ceil
from pathlib import Path
from statistics import median


def main():
    parser = argparse.ArgumentParser(description="Summarize an observe-mode demo CSV.")
    parser.add_argument("csv_file", type=Path)
    args = parser.parse_args()

    with args.csv_file.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    frames = [row for row in records if row["event"] == "frame" and row["enabled"] == "1"]
    if not frames:
        parser.exit(1, "No enabled decision frames in this run. Send start during the demo.\n")

    elapsed = float(frames[-1]["publish_monotonic_s"]) - float(frames[0]["publish_monotonic_s"])
    latency = sorted(float(row["latency_ms"]) for row in frames)
    print(f"Decision frames: {len(frames)}")
    if elapsed > 0:
        print(f"Effective decision rate: {(len(frames) - 1) / elapsed:.2f} Hz")
    print(f"Camera-to-decision latency: median {median(latency):.1f} ms, "
          f"P95 {latency[ceil(0.95 * len(latency)) - 1]:.1f} ms")
    print("Decisions:", dict(Counter(row["decision"] for row in frames)))
    stops = Counter(row["stop_reason"] for row in records if row["decision"] == "stop")
    print("Stop reasons (all events):", dict(stops))


if __name__ == "__main__":
    main()
