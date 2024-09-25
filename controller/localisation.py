#!/usr/bin/env python

import argparse
import sys

import cv2
import numpy as np
import yaml

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}

DEFAULT_SIZE = "DICT_4X4_50"
DEFAULT_SQUARE = 0.12  # metres
POINTS_IDS = [0, 1, 2, 3]  # IDs of the aruco markers at the corners of the pit
ROBOT_ID = 4  # ID for aruco marker on the robot


def main(args=None):
    dev = args.device

    cap = cv2.VideoCapture(dev)
    if not cap.isOpened():
        print("Could not open camera", file=sys.stderr)
        exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)
    print(f"fps: {cap.get(cv2.CAP_PROP_FPS)}")

    mtx, dist = get_cam_params(args.cam_params)

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args.size])
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    tf_wc = None
    marker_size = args.square
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

        corners, ids, rejected = detector.detectMarkers(frame)
        tf_cw = get_tf_cw(ids, corners, POINTS_IDS, marker_size, mtx, dist)
        if tf_cw is not None:
            tf_wc = np.linalg.inv(tf_cw)

        if tf_wc is None:
            continue


def get_cam_params(file: str) -> tuple[np.ndarray, np.ndarray]:
    """Return mtx, dist camera parameters from yaml file"""
    with open(file, "r") as f:
        params = yaml.safe_load(f)

    mtx_shape = (
        params["camera_matrix"]["rorobotics name for central controller serverws"],
        params["camera_matrix"]["cols"],
    )
    mtx = np.array(params["camera_matrix"]["data"])
    mtx = mtx.reshape(mtx_shape)

    dist_shape = (
        params["distortion_coefficients"]["rows"],
        params["distortion_coefficients"]["cols"],
    )
    dist = np.array(params["distortion_coefficients"]["data"])
    dist = dist.reshape(dist_shape)

    return mtx, dist


def get_tf_cw(
    ids: np.ndarray,
    corners: np.ndarray,
    point_ids: list[int],
    marker_size: float,
    mtx: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray | None:
    """
    Get transform to the origin.

    Params:
        ids: Aruco marker ids detected in the image.
        corners: The locations of the corners of the aruco markers in the image (listed in the same order as ids).
        point_ids: The ids of the aruco markers (top_left, top_right, bottom_left, bottom_right)
        marker_size: The side length of the aruco markers in metres.
        mtx: The camera intrinsics matrix.
        dist: The distortion coefficients for the camera.

    Returns:
        The transformation matrix of camera to world with origin at bottom_left tag (with z facing up out of the plane) or None
        if not enough markers were found.
    """
    marker_points = np.array(
        [
            [-marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, -marker_size / 2, 0],
            [-marker_size / 2, -marker_size / 2, 0],
        ],
        dtype=np.float32,
    )

    t_vecs = []
    res_ids = []

    for id, corner in zip(ids, corners):
        if id not in point_ids:
            continue

        success, r_vec, t_vec = cv2.solvePnP(
            marker_points, corner, mtx, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )

        if not success:
            continue

        t_vecs.append(t_vec)
        res_ids.append(id)

    t_vecs = np.array(t_vecs)
    if len(t_vecs) < 3 or point_ids[2] not in res_ids:
        # TODO: ways that do not require the origin
        # insufficient points to get the plane or origin not in points
        return None

    centroid = np.mean(t_vecs, axis=1)
    svd = np.linalg.svd(t_vecs - centroid)
    new_z = svd[0, :, -1]

    idx = res_ids.index(point_ids[2])
    # translation to world origin
    t = t_vecs[idx]

    if point_ids[3] in res_ids:
        idx = res_ids.index(point_ids[3])
        new_x = t_vecs[idx] - t
        new_x /= np.linalg.norm(new_x)
        new_y = np.cross(new_z, new_x)
    else:
        idx = res_ids.index(point_ids[0])
        new_y = t_vecs[idx] - t
        new_y /= np.linalg.norm(new_y)
        new_x = np.cross(new_y, new_z)

    R = np.column_stack((new_x, new_y, new_z))
    tf = np.eye(4)
    tf[:3, :3] = R
    tf[:3, 3] = t
    return tf


def get_robot_pos(
    tf_wc: np.ndarray,
    ids: np.ndarray,
    corners: np.ndarray,
    marker_size: float,
    mtx: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray | None:
    """
    Get the robot's pose.

    Params:
        tf_wc: Transformation matrix from world to camera frame.
        ids: Aruco marker ids detected in the image.
        corners: The locations of the corners of the aruco markers in the image (listed in the same order as ids).
        marker_size: The side length of the aruco markers in metres.
        mtx: The camera intrinsics matrix.
        dist: The distortion coefficients for the camera.

    Returns:
        The robot's pose in the world frame.
    """
    # TODO: will use multiple tags... This can be slightly easier with ROS2 if we use it
    if ROBOT_ID not in ids:
        return None

    marker_points = np.array(
        [
            [-marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, -marker_size / 2, 0],
            [-marker_size / 2, -marker_size / 2, 0],
        ],
        dtype=np.float32,
    )
    idx = np.where(ids == ROBOT_ID)[0][0]
    success, r_vec, t_vec = cv2.solvePnP(
        marker_points, corners[idx], mtx, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )

    if not success:
        return None

    tf_cr = np.eye(4)
    R = cv2.Rodrigues(r_vec)
    tf_cr[:3, :3] = R
    tf_cr[:3, 3] = t_vec

    return tf_wc @ tf_cr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("device")
    parser.add_argument("cam_params")
    parser.add_argument("--size", default=DEFAULT_SIZE)
    parser.add_argument("--square", type=float, default=DEFAULT_SQUARE)
    main(parser.parse_args())
