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
from path_planner_arc import PathPlannerArc
from path_planning_zigzag import PathPlannerZigZag
from pynput import keyboard
from utils.path_planning_visualiser import visualize_segments_zig_zag

DEFAULT_SIZE = "DICT_4X4_50"
WHEEL_RADIUS = 0.0396  # radius of wheel in m
RPM_TO_RAD_S = 2 * math.pi / 60

# CONSTANTS FOR THE PATH
COVERABLE_AREA_WIDTH = 1.400
COVERABLE_AREA_HEIGHT = 1.400
WAYPOINT_SPACING = 0.05
START_X = 0.300
START_Y = 0.300
SCOOP_WIDTH = 0.180
OVERLAP_PERCENTAGE = 20

# CONSTANTS FOR PI CONTROLLER
KP = 2
KI = 0.6

# MODE ENUM
WAITING_SIGNAL = 0
CONTINUE = 1
GO_TO_HIGH_GROUND = 2
DISPENSE_BEANS = 3
RETURN_TO_DEPOSIT_FAR = 4
RETURN_TO_DEPOSIT_NEAR = 5
RETURN_TO_POSITION = 6

# Set controller defaults
MAX_WHEEL_RPM = 100
MAX_SPEED = MAX_WHEEL_RPM * RPM_TO_RAD_S * WHEEL_RADIUS

# PIT CONSTANTS
PIT_MIN_X = 0
PIT_MAX_X = 2
PIT_MIN_Y = 0
PIT_MAX_Y = 2

OFFSET_TO_WALL = 0.1

# KEYBOARD HANDLER CONSTANTS
HIGH_GROUND_KEY = "h"
DELIVERY_REQUEST_KEY = "d"
DISPENSE_BEANS = "b"
START_KEY = "s"

# DIGGING FLAG
digging_flag = False

# GLOBAL KEYPRESS FLAGS
high_ground_request = False
return_to_delivery_point_request = False
dispense_beans_request = False
start_deployment_request = False


def main(args=None):
    # GLOBAL KEYPRESS FLAGS
    global high_ground_request
    global return_to_delivery_point_request
    global dispense_beans_request
    global start_deployment_request
    # Get Camera
    dev = args.device
    cap = BufferlessVideoCapture(dev)
    mtx, dist = get_cam_params(args.cam_params)

    # Setup localisation
    loc = Localisation(
        mtx, dist, ROBOT_MARKERS, marker_size=args.square, dict_type=args.size
    )

    look_ahead = 0.2

    # Start listener thread
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # Comms
    with open("comms_config.json") as f:
        data = json.load(f)

    ip = data["ip"]
    comms = Comms(ip)

    # Genearte the overall path
    ppc = PurePursuitController(look_ahead, 0.5 * MAX_SPEED, tol=0.025)

    # Generate heading controller
    hc = HeadingController(KP, KI)

    # Construct the path follower
    pf = PathFollower(ppc, hc, comms)

    # Construct the zigzag path class
    radial_planner = PathPlannerArc(0.3, 0.3, 0.18, 10, 1.4, WAYPOINT_SPACING)

    # Generate the zigzag path
    rad_path = radial_planner.generate_all_radial_segments()
    visualize_segments_zig_zag(rad_path)

    # Construct the main controller
    mc = MainController(rad_path, comms, pf)
    # mc.set_mode(CONTINUE)

    # Flag for digging
    global digging_flag
    prev_time = time.time()

    # TODO REMOVE
    temp_flag = False
    comms.send_container_request(False)

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

        # Let the main controller update pose
        mc.set_current_pose(pose)

        # Poll for mode interrupt here!
        if high_ground_request:
            print("Going to high ground")
            high_ground_request = False
            digging_flag = False
            mc.set_mode(GO_TO_HIGH_GROUND)
        elif start_deployment_request:
            print("Starting deployment")
            start_deployment_request = False
            digging_flag = False
            mc.set_mode(CONTINUE)
        elif return_to_delivery_point_request:
            print("Returning to delivery point")
            return_to_delivery_point_request = False
            digging_flag = False
            mc.set_mode(RETURN_TO_DEPOSIT_FAR)
        elif dispense_beans_request:
            print("Dispensing beans")
            dispense_beans_request = False
            digging_flag = False
            mc.set_mode(DISPENSE_BEANS)

        # if (time.time() - prev_time) > 30.0 and temp_flag == False:
        # mc.set_mode(GO_TO_HIGH_GROUND)
        # temp_flag = True

        action = np.array([0.0, 0.0])
        # Get the actions for the controller paths that are valuable
        current_mode = mc.get_mode()
        if current_mode != WAITING_SIGNAL:
            action = mc.mc_get_control_action()

            if action[0] == 0 and action[1] == 0:
                # Handle transitions if necessary

                if current_mode == RETURN_TO_POSITION:
                    digging_flag = True
                    mc.set_mode(CONTINUE)
                elif current_mode == RETURN_TO_DEPOSIT_FAR:
                    mc.set_mode(RETURN_TO_DEPOSIT_NEAR)
                elif (
                    current_mode == RETURN_TO_DEPOSIT_NEAR
                    or current_mode == GO_TO_HIGH_GROUND
                ):
                    mc.set_mode(WAITING_SIGNAL)
                elif current_mode == CONTINUE:
                    digging_flag = True
                    mc.set_mode(CONTINUE)
                continue

        if digging_flag == True and (
            -math.pi / 2
        ) < mc._path_follower._ppc.get_path_angle() < (0):
            # Send the scoop request
            if (time.time() - prev_time) > 3.0:
                time.sleep(0.05)
                comms.send_drive_request(0.0, 0.0)
                time.sleep(0.05)
                comms.send_scoop_request(True)
                time.sleep(3.0)
                comms.send_scoop_request(False)
                time.sleep(3.0)

                prev_time = time.time()
                continue

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


class PathFollower:
    def __init__(self, ppc: PurePursuitController, hc: HeadingController, comms: Comms):
        self._path = None
        self._ppc = ppc
        self._hc = hc
        self._mode = 0
        self._initial_turn = True
        self._comms = comms

    def set_path(self, path):
        self._path = path
        self._ppc.path = path
        self._initial_turn = True

    def get_path(self):
        return self._path

    def get_control_action(self, current_pose):
        global digging_flag

        # if self._initial_turn:
        #     digging_flag = False
        #
        #     action_turn = self._hc.get_control_action(current_pose[2], self._path[0][2])
        #     if action_turn != 0:
        #         return (0, action_turn)
        #
        #     digging_flag = True
        #     time.sleep(0.05)
        #     self._comms.send_scoop_request(False)
        #
        #     self._hc.reset()
        #     self._initial_turn = False

        action_straight = self._ppc.get_control_action(current_pose)
        if np.any(action_straight != 0):
            return action_straight

        self._hc.reset()
        return (0, 0)


class MainController:
    def __init__(self, path, comms: Comms, path_follower: PathFollower):
        self._mode = WAITING_SIGNAL
        self._overall_path = path
        self._current_segment = 0
        self._comms = comms
        self._path_follower = path_follower
        self._path_segment_idx = 0
        self._pose = (0.0, 0.0)
        self._stored_pose = (0.0, 0.0)
        self._container_position = (0.1, 0.1)
        self._has_been_moved = False

    def set_mode(self, mode):
        # Set path based on the current action
        global digging_flag
        self._path_follower._ppc.avg_speed = 0.4 * MAX_SPEED
        if mode == CONTINUE:
            # Go to start of segment logic
            if self._has_been_moved:
                # Perform opening and closing of door
                self._comms.send_container_request(True)
                time.sleep(5.0)

                self._comms.send_container_request(False)
                time.sleep(1.0)
                # mode = RETURN_TO_POSITION
                mode = CONTINUE
                self._has_been_moved = False
            else:
                # Logic for segmenting selection in normal occurance
                if self._path_segment_idx == len(self._overall_path):
                    self._mode = WAITING_SIGNAL
                    return
                else:
                    self._path_segment_idx += 1

            # Lower Scoop
            time.sleep(0.05)
            self._comms.send_scoop_request(False)

            # Set the path for the controller
            current_segment = self._overall_path[self._path_segment_idx - 1]
            if (-math.pi / 2) < current_segment[0][2] < (0):
                digging_flag = True
                time.sleep(0.05)
                self._comms.send_scoop_request(False)
            else:
                digging_flag = False
                time.sleep(0.05)
                self._comms.send_scoop_request(True)

            logger.info(f"Current segment: {current_segment}")
            self._path_follower.set_path(current_segment)

        elif mode == DISPENSE_BEANS:
            # Dispense beans, which is essentially just stopping the robot and going to waiting
            self._comms.send_drive_request(0, 0)

            # Change mode to be waiting for a signal
            mode = WAITING_SIGNAL

        elif mode == GO_TO_HIGH_GROUND:
            # Store the current pose
            self.set_stored_pose()
            self.set_has_been_moved()
            time.sleep(0.01)
            self._comms.send_scoop_request(True)
            digging_flag = False

            # Generate path to go to high ground, will be wrapped with desired location
            path = sand_snake_path(self.get_current_pose())
            self._path_follower.set_path(path)
        elif mode == RETURN_TO_DEPOSIT_FAR:
            # Store the current pose
            self.set_stored_pose()
            self.set_has_been_moved()
            time.sleep(0.05)
            self._comms.send_scoop_request(True)
            digging_flag = False

            path = generate_straight_line(
                self.get_current_pose(), self._container_position, WAYPOINT_SPACING
            )
            self._path_follower.set_path(path)
        elif mode == RETURN_TO_DEPOSIT_NEAR:
            # We need custom functionality that does not use either controller to slowly drift forwards
            path = generate_straight_line(
                self.get_current_pose(), (0.1, 0.1), WAYPOINT_SPACING
            )
            digging_flag = False
            # self._path_follower._ppc.avg_speed *= 0.5
            self._path_follower.set_path(path)
            pass

        if mode == RETURN_TO_POSITION:
            digging_flag = False
            time.sleep(0.05)
            self._comms.send_scoop_request(True)

            path = generate_straight_line(
                self.get_current_pose(), self.get_stored_pose(), WAYPOINT_SPACING
            )

            self._path_follower.set_path(path)

        self._mode = mode

    def mc_get_control_action(self):
        return self._path_follower.get_control_action(self.get_current_pose())

    def get_mode(self):
        return self._mode

    def set_stored_pose(self):
        self._stored_pose = self._pose

    def get_stored_pose(self):
        return self._stored_pose

    def set_current_pose(self, pose):
        self._pose = pose

    def get_current_pose(self):
        return self._pose

    def set_has_been_moved(self):
        self._has_been_moved = True

    def clear_has_been_moved(self):
        self._has_been_moved = False

    def get_has_been_moved(self):
        return self._has_been_moved


def generate_straight_line(start, stop, spacing=0.05):
    # Generate the points for the straight line
    num_lines = round(
        math.sqrt(abs(stop[0] - start[0]) ** 2 + abs(stop[1] - start[1]) ** 2) / spacing
    )

    # Generate the x and y points
    x_points = np.linspace(start[0], stop[0], num_lines)
    y_points = np.linspace(start[1], stop[1], num_lines)

    # If the points are empty, append the end position
    if len(x_points) == 0 or len(y_points) == 0:
        x_points = np.array([stop[0]])
        y_points = np.array([stop[1]])

    heading = 0
    if stop[0] == start[0]:
        if start[1] > stop[1]:
            heading = math.radians(180)
        else:
            heading = 0
    else:
        heading = math.atan2((stop[1] - start[1]), (stop[0] - start[0]))

        # Adjust to be within the correct range
        heading -= math.pi / 2

    # Combine the points so that it is (x, y, heading)
    points = np.column_stack((x_points, y_points, np.ones_like(x_points) * heading))

    return points


def sand_snake_path(pose):
    """
    Gives the path that the robot should take if the sand snake signal is given. Goes to the closest wall.

    Params:
        pose: The current pose of the robot. (x,y,pi)

    Returns:

    """
    x_from_centre = abs(pose[0] - 1)
    y_from_centre = abs(pose[1] - 1)

    start_point = pose
    end_point = None
    if x_from_centre > y_from_centre:
        if pose[0] > 1:
            end_point = (PIT_MAX_X - OFFSET_TO_WALL, pose[1], math.pi / 2)
        else:
            end_point = (OFFSET_TO_WALL, pose[1], -math.pi / 2)
    else:
        if pose[1] > 1:
            end_point = (pose[0], PIT_MAX_Y - OFFSET_TO_WALL, 0)
        else:
            end_point = (pose[0], OFFSET_TO_WALL, math.pi)
    return generate_straight_line(start_point, end_point)


def on_release(key):
    if key == keyboard.Key.esc:
        # Stop listener
        return False


def on_press(key):
    global high_ground_request, start_deployment_request
    global return_to_delivery_point_request, dispense_beans_request

    try:
        if key.char == HIGH_GROUND_KEY:
            high_ground_request = True
        elif key.char == START_KEY:
            start_deployment_request = True
        elif key.char == DELIVERY_REQUEST_KEY:
            return_to_delivery_point_request = True
        elif key.char == DISPENSE_BEANS:
            dispense_beans_request = True
        else:
            print("unknown key")

    except Exception as e:
        print("error:", e)
        # print(f'Special key {key} pressed')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("device")
    parser.add_argument("cam_params")
    parser.add_argument("--size", default=DEFAULT_SIZE)
    parser.add_argument("--square", type=float, default=0.12)
    main(parser.parse_args())
