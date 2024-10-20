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
WAITING_SIGNAL = 0
CONTINUE = 1
GO_TO_HIGH_GROUND = 2
DISPENSE_BEANS = 3
RETURN_TO_DEPOSIT_FAR = 4
RETURN_TO_DEPOSIT_NEAR = 5

# CONTROLLER ENUM
STRAIGHT = 100
TURN = 101

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
        action_heading = pose[2]


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

class PathFollower:
    def __init__(self, ppc, hc):
        self._path = None
        self._ppc = ppc
        self._hc = hc
        self._mode = 0
        self._initial_turn = True

    def set_path(self, path):
        self._path = path
        self._ppc.path = path
        self._initial_turn = True

    def get_path(self):
        return self._path
    
    def get_control_action(self, current_pose):
        if self._initial_turn:
            action_turn = self._hc.get_control_action(current_pose[2], self._path[0][2])
            if action_turn != 0:
                return (0, action_turn)
            
            self._initial_turn = False

        action_straight = self._ppc.get_control_action(current_pose)
        if np.any(action_straight) != 0:
            return action_straight
        
        return (0, 0)
        

class MainController:
    def __init__(self, path, comms : Comms, path_follower : PathFollower):
        self._mode = WAITING_SIGNAL
        self._overall_path = path
        self._current_segment = 0
        self._comms = comms
        self._path_follower = path_follower
        self._path_segment_idx = 0
        self._stored_pose = (0, 0)

    def set_mode(self, mode, pose):
        # Set path based on the current action
        if mode == CONTINUE:
            # Perform opening and closing of door
            self._comms.send_container_request(True)
            time.sleep(5.0)

            self._comms.send_container_request(False)
            time.sleep(1.0)

            # Go to start of segment logic


            # Logic for segmenting selection in normal occurance
            if self._path_segment_idx == len(self._overall_path):
                self._mode = WAITING_SIGNAL
                return
            else:
                self._path_segment_idx += 1

            # Set the path for the controller
            current_segment = self._overall_path[self._path_segment_idx - 1]
            self._path_follower.set_path(current_segment)
            
        elif mode == DISPENSE_BEANS:
            # Dispense beans, which is essentially just stopping the robot and going to waiting
            self._comms.send_drive_request(0, 0)
            
            # Change mode to be waiting for a signal
            mode = WAITING_SIGNAL 

        elif mode == GO_TO_HIGH_GROUND:
            # Generate path to go to high ground, will be wrapped with desired location
            
            path = generate_straight_line(pose, (0.0, 0.0))
            self._path_follower.set_path(path)
        elif mode == RETURN_TO_DEPOSIT_FAR:
            path = generate_straight_line(pose, (0.0, 0.0)) 
            self._path_follower.set_path(path)
        elif mode == RETURN_TO_DEPOSIT_NEAR:
            path = generate_straight_line(pose, (0.0, 0.0)) 
            self._path_follower.set_path(path) 

        else:
            pass
        
        self._mode = mode
    
    def get_mode(self):
        return self._mode
    

def generate_straight_line(start, stop, spacing = 0.05):
    # Generate the points for the straight line
    num_lines = round(math.sqrt(abs(stop[0] - start[0]) ** 2 + abs(stop[1] - start[1]) ** 2) / spacing)

    # Generate the x and y points
    x_points = np.linspace(start[0], stop[0], num_lines)
    y_points = np.linspace(start[1], stop[1], num_lines)
    heading = math.tan((stop[1] - start[1])/(stop[0] - start[0]))

    # Combine the points so that it is (x, y, heading)
    points = np.column_stack((x_points, y_points, np.ones(num_lines) * heading))

    return points