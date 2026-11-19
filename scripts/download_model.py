#!/usr/bin/env python3
import argparse
from pathlib import Path

from roban_person_follower.model import default_model_path, download_model


def main():
    parser = argparse.ArgumentParser(description="Download and verify the official YOLOX-Nano weights.")
    parser.add_argument("--output", type=Path, default=default_model_path())
    args = parser.parse_args()
    print(download_model(args.output))


if __name__ == "__main__":
    main()
