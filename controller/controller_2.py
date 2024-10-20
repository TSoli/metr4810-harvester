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

# MODE ENUM
CONTINUE_SEQUENCE_START = 0
CONTINUE_SEQUENCE_TURN = 1
CONTINUE_SEQUENCE_STRAIGHT = 2
RETURN_TO_DEPOSIT = 3
DISPENSE_BEANS = 4
GO_TO_HIGH_GROUND = 5
WAITING_SIGNAL = 6

# CONTROLLER ENUM
STRAIGHT = 100
TURN = 101

# Global variabls
mode = 0
controller = 100

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

    mode = 0
    next_segment_flag = 1

    while True:
        # Localise the robot
        frame = cap.read()

        # Localise the initial points and get the current heading
        start_loc = time.time()
        tf_wr = loc.localise(frame)
        logger.info(f"Localisation took: {1e3 * (time.time() - start_loc)}ms")
        if tf_wr is None:
            continue
        
        # Extract the pose
        start_plan = time.time()
        pose = np.array(extract_pose_from_transform(tf_wr))

        # Poll for mode interrupt here!

        # Set path based on the current action
        if mode == RETURN_TO_DEPOSIT:
            # Perform return to deposit signal and wait for signal when finished

            mode = WAITING_SIGNAL
        elif mode == DISPENSE_BEANS:
            # Dispense beans, which is essentially just stopping the robot and going to waiting

            # 
            
            mode = WAITING_SIGNAL
            
        elif mode == GO_TO_HIGH_GROUND:
            # Perform go to high ground sequence and wait for signal when finished
            action_values = [0, 0]

            mode = WAITING_SIGNAL
            
        elif mode == CONTINUE_SEQUENCE_START:
            # Perform opening and closing of door sequence, need to import and run function
            
            # Continue to previous path, going to current segment
            next_segment_flag = 0
            mode = CONTINUE_SEQUENCE_START

        elif mode == CONTINUE_SEQUENCE_STRAIGHT:
            # Logic for segmenting selection in normal occurance
            if path_segment_idx == number_of_path_segments and next_segment_flag == 1:
                break
            else:
                path_segment_idx += 1

            # Ensure we go to next segment
            next_segment_flag = 1

            # Set the path for the controller
            current_segment = zig_zag_path[path_segment_idx - 1]
            ppc.path = current_segment       

            # Go along the straight line
            path_complete = straight_line_movement(cap, loc, hc, ppc, comms, current_segment, args, mtx, dist)
            if not path_complete:
                # Interupt occured
                interupt_flag = 1
                return 
            
            # Move on to the next segment after a pause
            time.sleep(0.5)

        elif mode == WAITING_SIGNAL:
            # Waiting for signal
            time.sleep(0.1)
        else:
            time.sleep(0.1)
        
        if mode == STRAIGH:
        
        # Get the control action and check to see if you are at the end of the path
        action = hc.get_control_action(pose[2], current_segment[0][2])
        if abs(pose[2] - current_segment[0][2]) < hc.get_tolerance():
            mode = WAITING_SIGNAL
            break

        # Log the timining of the plan
        start_comms = time.time()
        logger.info(f"Plan took {1e3 * (start_comms - start_plan)}")
        comms.send_drive_request(0, action)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("device")
    parser.add_argument("cam_params")
    parser.add_argument("--size", default=DEFAULT_SIZE)
    parser.add_argument("--square", type=float, default=0.12)
    main(parser.parse_args())
