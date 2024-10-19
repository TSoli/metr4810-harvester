import argparse
import json
import math
import time

import cv2
import numpy as np
from comms import Comms
from localisation import (ROBOT_MARKERS, BufferlessVideoCapture, Localisation,
                          draw_axes, extract_pose_from_transform,
                          get_cam_params)
from logger import logger
from path_following import HeadingController, PurePursuitController

DEFAULT_SIZE = "DICT_4X4_50"
WHEEL_RADIUS = 0.0396  # radius of wheel in m
RPM_TO_RAD_S = 2 * math.pi / 60


def main(args=None):
    dev = args.device

    cap = BufferlessVideoCapture(dev)

    mtx, dist = get_cam_params(args.cam_params)

    # Setup localisation
    loc = Localisation(
        mtx, dist, ROBOT_MARKERS, marker_size=args.square, dict_type=args.size
    )

    # controller
    max_wheel_rpm = 100
    max_speed = max_wheel_rpm * RPM_TO_RAD_S * WHEEL_RADIUS
    look_ahead = 0.2

    ppc = PurePursuitController(look_ahead, 0.4 * max_speed, tol=0.30)
    y_p = np.linspace(0.4, 1.6, 10)
    x_p = np.ones_like(y_p) * 0.4
    theta_p = np.zeros_like(y_p)
    path = np.column_stack((x_p, y_p, theta_p))
    ppc.path = path

    # comms
    with open("comms_config.json") as f:
        data = json.load(f)

    ip = data["ip"]
    comms = Comms(ip)
    # put the scoop up
    comms.send_scoop_request(True)
    time.sleep(3)

    while True:
        frame = cap.read()

        tf_wr = loc.localise(frame)
        if tf_wr is None:
            continue

        pose = np.array(extract_pose_from_transform(tf_wr))

        action = ppc.get_control_action(pose)
        if np.all(action == 0):
            break

        comms.send_drive_request(action[0], action[1])

        tf_cw = loc.tf_cw
        draw_axes(frame, args.square, tf_cw, mtx, dist)

        tf_cr = tf_cw @ tf_wr
        draw_axes(frame, args.square, tf_cr, mtx, dist)

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            exit(0)

    comms.send_drive_request(0, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("device")
    parser.add_argument("cam_params")
    parser.add_argument("--size", default=DEFAULT_SIZE)
    parser.add_argument("--square", type=float, default=0.12)
    main(parser.parse_args())
