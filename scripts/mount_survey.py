#!/usr/bin/env python3
"""Estimate neutral optical pitch from the floor and a measured lens height."""

import argparse
from math import atan2, degrees
import sys

import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, Image

from roban_person_follower.image import decode_image


def sample_floor(depth, p, measured_height):
    height, width = depth.shape
    rows, cols = np.mgrid[int(.52*height):height:4, int(.08*width):int(.92*width):4]
    z = depth[rows, cols].astype(float)
    if depth.dtype == np.uint16:
        z *= .001
    keep = np.isfinite(z) & (z >= .25) & (z <= 3)
    pixels = cols[keep]
    z = z[keep]
    if len(z) < 180 or p[0] <= 0 or p[5] <= 0:
        raise ValueError("Too few usable floor points")
    points = np.column_stack(((pixels-p[2])*z/p[0], (rows[keep]-p[6])*z/p[5], z))

    rng = np.random.default_rng(159)
    triples = points[rng.integers(0, len(points), size=(480,3))]
    normals = np.cross(triples[:,1]-triples[:,0], triples[:,2]-triples[:,0])
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:,None], 1e-9)
    normals *= np.where(normals[:,1] >= 0, 1, -1)[:,None]
    offsets = (normals*triples[:,0]).sum(axis=1)
    valid = ((normals[:,1]>.72) & (abs(normals[:,0])<.08)
             & (abs(offsets-measured_height)<.06))
    if not valid.any():
        raise ValueError("No level-floor candidate agrees with the measured lens height")
    fits = abs(points @ normals[valid].T - offsets[valid]) < .02
    inliers = fits[:, np.argmax(fits.sum(axis=0))]
    if inliers.sum() < 120 or np.ptp(pixels[inliers]) < .42 * width:
        raise ValueError("The floor plane lacks usable coverage across the image")
    cloud = points[inliers]
    center = cloud.mean(axis=0)
    normal = np.linalg.svd(cloud-center, full_matrices=False)[2][-1]
    if normal[1] < 0:
        normal *= -1
    floor_height = float(center @ normal)
    roll = degrees(atan2(normal[0], normal[1]))
    pitch = degrees(atan2(normal[2], normal[1]))
    if abs(floor_height-measured_height) >= .06 or abs(roll)>3 or not -15<=pitch<=45:
        raise ValueError("The fitted floor disagrees with the mount limits")
    return floor_height, pitch, roll, int(inliers.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=float, required=True, help="Measured lens height above level floor in meters")
    parser.add_argument("--namespace", default="/person_follower")
    parser.add_argument("--depth-image", help="Override the default registered image topic")
    parser.add_argument("--depth-info", help="CameraInfo for the selected depth image topic")
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])
    if not .15 <= args.height <= 1.2:
        parser.error("Enter the measured lens height in meters")
    if bool(args.depth_image) != bool(args.depth_info):
        parser.error("Specify both --depth-image and --depth-info when overriding depth topics")
    ns = args.namespace.rstrip("/")
    image_topic = args.depth_image or ns+"/depth_registered/image_rect"
    info_topic = args.depth_info or ns+"/depth_registered/camera_info"
    rospy.init_node("mount_survey", anonymous=True)
    info = rospy.wait_for_message(info_topic, CameraInfo, timeout=5)
    results = []
    for _ in range(3):
        msg = rospy.wait_for_message(image_topic, Image, timeout=5)
        if msg.header.frame_id != info.header.frame_id:
            raise RuntimeError("Depth and calibration frame do not match")
        try:
            result = sample_floor(decode_image(msg), info.P, args.height)
        except ValueError as error:
            parser.exit(2, str(error)+"\n")
        results.append(result)
    results = np.asarray(results)
    if np.ptp(results[:,0]) > .035 or np.ptp(results[:,1]) > 3:
        parser.exit(2, "Floor estimates varied between frames\n")
    height, pitch, roll, count = np.median(results, axis=0)
    print(f"Measured lens height {args.height:.3f} m; depth plane {height:.3f} m; "
          f"pitch {pitch:.2f} deg down; roll {roll:.2f} deg; {count:.0f} inliers")


if __name__ == "__main__":
    main()
