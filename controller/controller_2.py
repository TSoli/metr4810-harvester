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
from path_following import HeadingController, PurePursuitController, straight_line_movement
from path_planning_zigzag import PathPlannerZigZag

DEFAULT_SIZE = "DICT_4X4_50"
WHEEL_RADIUS = 0.0396  # radius of wheel in m
RPM_TO_RAD_S = 2 * math.pi / 60

# CONSTANTS FOR THE PATH
COVERABLE_AREA_WIDTH = 1.900
COVERABLE_AREA_HEIGHT = 1.900
WAYPOINT_SPACING = 0.100
START_X = 0.100
START_Y = 0.100
SCOOP_WIDTH = 0.180
OVERLAP_PERCENTAGE = 20

# CONSTANTS FOR PI CONTROLLER
KP = 0.1
KI = 0.01

def main(args=None):
    # Get Camera
    dev = args.device
    cap = BufferlessVideoCapture(dev)
    mtx, dist = get_cam_params(args.cam_params)

    # Setup localisation
    loc = Localisation(
        mtx, dist, ROBOT_MARKERS, marker_size=args.square, dict_type=args.size
    )

    # Set controller defaults
    max_wheel_rpm = 100
    max_speed = max_wheel_rpm * RPM_TO_RAD_S * WHEEL_RADIUS
    look_ahead = 0.2

    # Comms
    with open("comms_config.json") as f:
        data = json.load(f)

    ip = data["ip"]
    comms = Comms(ip)

    # Genearte the overall path
    ppc = PurePursuitController(look_ahead, 0.4 * max_speed, tol=0.05)
    
    # Generate heading controller
    hc = HeadingController(KP, KI)

    # Construct the zigzag path class
    zig_zag_planner = PathPlannerZigZag(
        SCOOP_WIDTH, OVERLAP_PERCENTAGE, WAYPOINT_SPACING, 
        START_X, START_Y, COVERABLE_AREA_WIDTH, COVERABLE_AREA_HEIGHT
    )

    # Generate the zigzag path
    zig_zag_path = zig_zag_planner.generate_zigzag_path()  
    number_of_path_segments = zig_zag_planner.get_number_of_segments()
    path_segment_idx = 0

    while True:
        # Logic for segmenting selection
        if path_segment_idx == number_of_path_segments:
            break
        else:
            path_segment_idx += 1

        # Set the path for the controller
        current_segment = zig_zag_path[path_segment_idx - 1]
        ppc.path = current_segment       

        # Go along the straight line
        path_complete = straight_line_movement(cap, loc, hc, ppc, comms, current_segment, args, mtx, dist)
        if not path_complete:
            # Log error
            logger.error("Error in path following")
            return 
        
        # Move on to the next segment after a pause
        time.sleep(0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("device")
    parser.add_argument("cam_params")
    parser.add_argument("--size", default=DEFAULT_SIZE)
    parser.add_argument("--square", type=float, default=0.12)
    main(parser.parse_args())
