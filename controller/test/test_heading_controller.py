import argparse
import json
import os
import sys

import cv2
import numpy as np

parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_directory)

from comms import Comms
from localisation import (ROBOT_MARKERS, Localisation,
                          extract_pose_from_transform, get_cam_params)
from logger import logger
from path_following import HeadingController

DEFAULT_SIZE = "DICT_4X4_50"


def main(args=None):
    dev = args.device

    cap = cv2.VideoCapture(dev)
    if not cap.isOpened():
        logger.warning("Could not open camera")
        exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)

    mtx, dist = get_cam_params(args.cam_params)

    # Setup localisation
    loc = Localisation(
        mtx, dist, ROBOT_MARKERS, marker_size=args.square, dict_type=args.size
    )

    hc = HeadingController(0.5, 0.005)

    # comms
    with open(os.path.join("..", "comms_config.json")) as f:
        data = json.load(f)

    ip = data["ip"]
    comms = Comms(ip)
    # put the scoop up
    comms.send_scoop_request(True)

    goal_headings = [np.pi, 0, -np.pi / 2]
    for goal in goal_headings:
        logger.info(f"Attempting to reach heading: {goal:.2f}")
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            tf_wr = loc.localise(frame)
            if tf_wr is None:
                continue

            _, _, heading = np.array(extract_pose_from_transform(tf_wr))
            w = hc.get_control_action(heading, goal)
            if w == 0:
                break

            comms.send_drive_request(0, w)

        logger.info("Goal Reached!")
        comms.send_drive_request(0, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("device")
    parser.add_argument("cam_params")
    parser.add_argument("--size", default=DEFAULT_SIZE)
    parser.add_argument("--square", type=float, default=0.12)
    main(parser.parse_args())
