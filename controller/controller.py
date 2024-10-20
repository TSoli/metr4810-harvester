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

    ppc = PurePursuitController(look_ahead, 0.4 * max_speed, tol=0.05)
    y_p = np.concat(
        (
            np.linspace(0.4, 1.6, 10),
            np.ones(10) * 1.6,
            np.linspace(1.6, 0.4, 10),
            # np.ones(10) * 0.4,
        )
    )
    x_p = np.concat(
        (
            np.ones(10) * 0.4,
            np.linspace(0.4, 1.6, 10),
            np.ones(10) * 1.6,
            # np.linspace(1.6, 0.4, 10),
        )
    )
    theta_p = np.zeros_like(y_p)
    path = np.column_stack((x_p, y_p, theta_p))

    ppc.path = path

    # comms
    with open("comms_config.json") as f:
        data = json.load(f)

    ip = data["ip"]
    comms = Comms(ip)
    # put the scoop up
    # comms.send_scoop_request(True)

    while True:
        frame = cap.read()

        # Localise the initial points
        start_loc = time.time()
        tf_wr = loc.localise(frame)
        logger.info(f"Localisation took: {1e3 * (time.time() - start_loc)}ms")
        if tf_wr is None:
            continue
        
        # Extract the pose
        start_plan = time.time()
        pose = np.array(extract_pose_from_transform(tf_wr))

        # Get the control action and check to see if you are at the end of the path
        action = ppc.get_control_action(pose)
        if np.all(action == 0):
            break

        # Log the timining of the plan
        start_comms = time.time()
        logger.info(f"Plan took {1e3 * (start_comms - start_plan)}")
        comms.send_drive_request(action[0], action[1])
        start_draw = time.time()
        logger.info(f"Comms took {1e3 * (start_draw - start_comms)}")

        # Draw axis for the corners
        tf_cw = loc.tf_cw
        draw_axes(frame, args.square, tf_cw, mtx, dist)

        # Draw the axis for the robot
        tf_cr = tf_cw @ tf_wr
        draw_axes(frame, args.square, tf_cr, mtx, dist)

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            exit(0)

        logger.info(f"Draw took {1e3 * (time.time() - start_draw)}")
        time.sleep(0.05)

    comms.send_drive_request(0, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("device")
    parser.add_argument("cam_params")
    parser.add_argument("--size", default=DEFAULT_SIZE)
    parser.add_argument("--square", type=float, default=0.12)
    main(parser.parse_args())
