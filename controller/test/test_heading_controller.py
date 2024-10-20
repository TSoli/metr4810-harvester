import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_directory)

from comms import Comms
from localisation import (ROBOT_MARKERS, BufferlessVideoCapture, Localisation,
                          extract_pose_from_transform, get_cam_params)
from logger import logger
from path_following import HeadingController

DEFAULT_SIZE = "DICT_4X4_50"


def main(args=None):
    dev = args.device

    cap = BufferlessVideoCapture(dev)

    mtx, dist = get_cam_params(args.cam_params)

    # Setup localisation
    loc = Localisation(
        mtx, dist, ROBOT_MARKERS, marker_size=args.square, dict_type=args.size
    )

    hc = HeadingController(1, 0.4)

    # comms
    with open(os.path.join("..", "comms_config.json")) as f:
        data = json.load(f)

    ip = data["ip"]
    comms = Comms(ip)
    # put the scoop up
    # comms.send_scoop_request(True)

    goal_headings = [-np.pi / 4, 0, np.pi, -np.pi / 2]
    for goal in goal_headings:
        logger.info(f"Attempting to reach heading: {goal:.2f}")
        hc.reset()
        while True:
            frame = cap.read()

            tf_wr = loc.localise(frame)
            if tf_wr is None:
                continue

            _, _, heading = np.array(extract_pose_from_transform(tf_wr))
            w = hc.get_control_action(heading, goal)
            if w == 0:
                break

            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break

            comms.send_drive_request(0.0, w)
            time.sleep(0.02)

        logger.info("Goal Reached!")
        comms.send_drive_request(0.0, 0.0)
        comms.send_drive_request(0.0, 0.0)
        comms.send_drive_request(0.0, 0.0)
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("device")
    parser.add_argument("cam_params")
    parser.add_argument("--size", default=DEFAULT_SIZE)
    parser.add_argument("--square", type=float, default=0.12)
    main(parser.parse_args())
