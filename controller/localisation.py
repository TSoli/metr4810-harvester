import argparse
import queue
import sys
import threading
import queue
import threading
from collections.abc import Sequence

import cv2
import numpy as np
import numpy.typing as npt
import numpy.typing as npt
import yaml
from logger import logger
from logger import logger

DEFAULT_SIZE = "DICT_4X4_50"
ROBOT_MARKERS = {
    # TOP MARKER
    10: np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, -0.055],
            [0, 0, 1, 0.364],
            [0, 0, 1, 0.364],
            [0, 0, 0, 1],
        ],
        dtype=float,
    ),
    # FRONT
    6: np.array(
        [
            [-1, 0, 0, 0],
            [0, 0, 1, 0.038],
            [0, 1, 0, 0.317],
            [0, 1, 0, 0.317],
            [0, 0, 0, 1],
        ],
        dtype=float,
    ),
    # BACK
    4: np.array(
        [
            [1, 0, 0, 0],
            [0, 0, -1, -0.148],
            [0, 1, 0, 0.272],
            [0, 1, 0, 0.272],
            [0, 0, 0, 1],
        ],
        dtype=float,
    ),
    # LEFT TOP
    5: np.array(
    5: np.array(
        [
            [0, 0, -1, -0.103],
            [-1, 0, 0, -0.055],
            [0, 1, 0, 0.282],
            [0, 1, 0, 0.282],
            [0, 0, 0, 1],
        ],
        dtype=float,
    ),
    # LEFT BOTTOM
    8: np.array(
    8: np.array(
        [
            [0, 0, -1, -0.103],
            [-1, 0, 0, -0.055],
            [0, 1, 0, 0.112],
            [0, 1, 0, 0.112],
            [0, 0, 0, 1],
        ],
        dtype=float,
    ),
    # RIGHT TOP
    7: np.array(
    7: np.array(
        [
            [0, 0, 1, 0.103],
            [1, 0, 0, -0.055],
            [0, 1, 0, 0.282],
            [0, 1, 0, 0.282],
            [0, 0, 0, 1],
        ],
        dtype=float,
    ),
    # RIGHT BOTTOM
    9: np.array(
    9: np.array(
        [
            [0, 0, 1, 0.103],
            [1, 0, 0, -0.055],
            [0, 1, 0, 0.112],
            [0, 0, 0, 1],
        ],
        dtype=float,
    ),
}

WORLD_TO_CORNERS = {
    0: np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 2.0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    ),
    1: np.array(
        [
            [1, 0, 0, 2.0],
            [0, 1, 0, 2.0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    ),
    2: np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    ),
    3: np.array(
        [
            [1, 0, 0, 2.0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    ),
}

CORNER_MARKERS = {
    0: np.array(
        [
            [0, 1, 0, -0.085],
            [-1, 0, 0, -0.215],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    ),
    1: np.array(
        [
            [0, 1, 0, 0.215],
            [-1, 0, 0, 0.085],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    ),
    2: np.array(
        [
            [-1, 0, 0, -0.085],
            [0, -1, 0, -0.175],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    ),
    3: np.array(
        [
            [-1, 0, 0, 0.085],
            [0, -1, 0, -0.175],
            [0, 0, 1, 0],
            [0, 1, 0, 0.112],
            [0, 0, 0, 1],
        ],
        dtype=float,
    ),
}

WORLD_TO_CORNERS = {
    0: np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 2.0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    ),
    1: np.array(
        [
            [1, 0, 0, 2.0],
            [0, 1, 0, 2.0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    ),
    2: np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    ),
    3: np.array(
        [
            [1, 0, 0, 2.0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    ),
}

CORNER_MARKERS = {
    0: np.array(
        [
            [0, -1, 0, -0.085],
            [1, 0, 0, -0.215],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    ),
    1: np.array(
        [
            [0, 1, 0, 0.085],
            [-1, 0, 0, -0.215],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    ),
    2: np.array(
        [
            [-1, 0, 0, -0.085],
            [0, -1, 0, -0.175],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    ),
    3: np.array(
        [
            [-1, 0, 0, 0.085],
            [0, -1, 0, -0.175],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    ),
}


# bufferless VideoCapture
class BufferlessVideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        if not self.cap.isOpened():
            logger.critical("Could not open camera")
            exit(1)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()


# bufferless VideoCapture
class BufferlessVideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        if not self.cap.isOpened():
            logger.critical("Could not open camera")
            exit(1)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()


class Localisation:
    """Localises the Robot given camera frames"""

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

    def __init__(
        self,
        mtx: np.ndarray,
        dist: np.ndarray,
        robot_markers: dict[int, np.ndarray],
        marker_size: float = 0.12,
        dict_type: str = "DICT_4X4_50",
        corner_ids: dict[int, np.ndarray] = CORNER_MARKERS,
        corner_ids: dict[int, np.ndarray] = CORNER_MARKERS,
        world_to_corners: dict[int, np.ndarray] = WORLD_TO_CORNERS,
    ) -> None:
        """
        Params:
            mtx, dist: Camera calibration parameters (see https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
            robot_markers: Map from aruco marker id to transformation matrix of
                robot frame to marker frame.
            marker_size: Side length of aruco markers in metres.
            dict_type: The aruco dictionary to use.
            corner_ids: The ids for the corners of the pit in order top left, top right,
                bottom left, bottom right as looking from above.
        """
        self._mtx = mtx
        self._dist = dist
        self._robot_markers = robot_markers
        self._marker_size = marker_size
        self._marker_points = np.array(
            [
                [-marker_size / 2, marker_size / 2, 0],
                [marker_size / 2, marker_size / 2, 0],
                [marker_size / 2, -marker_size / 2, 0],
                [-marker_size / 2, -marker_size / 2, 0],
            ],
            dtype=np.float32,
        )
        self._marker_pos = {}
        self._corner_ids = corner_ids
        self._world_to_corners = world_to_corners
        self._frame = None
        self._tf_wr = None
        self._tf_cw = None
        self._prev_rvec = None
        self._prev_tvec = None

        aruco_dict = cv2.aruco.getPredefinedDictionary(
            Localisation.ARUCO_DICT[dict_type]
        )
        aruco_params = cv2.aruco.DetectorParameters()
        aruco_params.minMarkerPerimeterRate = 8 / (4 * 1e3 * marker_size)
        self._aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    @property
    def tf_cw(self) -> np.ndarray | None:
        """Get the transform from the camera to the world frame"""
        return self._tf_cw

    @property
    def tf_wr(self) -> np.ndarray | None:
        """
        Get the transform from the world frame (bottom left of the pit) to the
        robot.
        """
        return self._tf_wr

    def localise(self, img: cv2.typing.MatLike) -> np.ndarray | None:
        """
        Localise the robot in the world frame.

        Params:
            img: The image to localise from.

        Returns:
            The transformation matrix from the world frame (bottom left corner of the pit)
            to the robot or None if it cannot be computed.
        """
        if not self.add_img(img):
            return None

        return self._tf_wr

    def add_img(self, img: cv2.typing.MatLike) -> bool:
        """
        Add an image to localise from.

        Params:
            img: The image to localise from.

        Returns:
            True if the robot localisation was successful.
        """
        self._frame = img
        corners, ids, rejected = self._aruco_detector.detectMarkers(img)
        if ids is None:
            return False

        updated_tf_cw = self._update_tf_cw(ids, corners)
        updated_tf_wr = self._update_tf_wr(ids, corners)
        return updated_tf_cw and updated_tf_wr

    def _update_tf_cw(
        self, ids: cv2.typing.MatLike, corners: Sequence[cv2.typing.MatLike]
    ) -> bool:
        """
        Update the transformation matrix from camera to world frame.

        Params:
            ids: The ids of the aruco markers detected in an image.
            corners: The corresponding corner locations of the aruco markers in the
                image.

        Returns:
            True if tf_cw was updated successfully.
        """

        res_t_vecs = []
        res_ids = []

        idx = np.isin(ids, list(self._corner_ids.keys()))
        for id, corner in zip(ids[idx], np.array(corners)[idx]):
        idx = np.isin(ids, list(self._corner_ids.keys()))

        corners = np.array(corners)
        corner_points = []
        for id, corner in zip(ids[idx], corners[idx]):
            # Add all of the points detected on the robot in the robot frame
            marker_points_homo = np.column_stack(
                (self._marker_points, np.ones(self._marker_points.shape[0]))
            )
            points = (
                self._world_to_corners[id] @ self._corner_ids[id] @ marker_points_homo.T
            ).T[:, :-1]
            corner_points.append(points)

        if len(corner_points) < 1:
            logger.warning("No corners detected")
            return False

        corner_points = np.vstack(corner_points)
        if self._prev_rvec is not None and self._prev_tvec is not None:
            success, r_vec, t_vec = cv2.solvePnP(
                corner_points,
                np.vstack(corners[idx]),
                self._mtx,
                self._dist,
                rvec=self._prev_rvec,
                tvec=self._prev_tvec,
                flags=cv2.SOLVEPNP_ITERATIVE,
                useExtrinsicGuess=True,
            )
        else:
            success, r_vec, t_vec = cv2.solvePnP(
                corner_points,
                np.vstack(corners[idx]),
                self._mtx,
                self._dist,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            if not success:
                continue

            tf = vecs_to_tf(r_vec, t_vec)
            r_vec, t_vec = tf_to_vecs(tf @ self._corner_ids[id])

            res_t_vecs.append(t_vec)
            res_ids.append(id)

        if len(corner_points) < 1:
            logger.warning("No corners detected")
            return False

        t_vecs = np.column_stack(res_t_vecs)
        # TODO: Magic number is id of tag
        bottom_left_id = 2
        if t_vecs.shape[1] < 3 or bottom_left_id not in res_ids:
            # The bottom left marker was not detected or not enough points
            # to estimate the pit plane
            # TODO: ways that do not require the origin
            # insufficient points to get the plane or origin not in points
        self._prev_rvec = r_vec
        self._prev_tvec = t_vec

        if not success:
            logger.warning("Failed to estimate origin")
            return False

        tf_cw = np.eye(4)
        R, _ = cv2.Rodrigues(r_vec)
        tf_cw[:3, :3] = R
        tf_cw[:3, 3] = t_vec.T
        self._tf_cw = tf_cw

        for id in ids[idx]:
            draw_axes(
                self._frame,
                0.15,
                tf_cw @ self._world_to_corners[id] @ self._corner_ids[id],
                self._mtx,
                self._dist,
            )
        return True

    def _update_tf_wr(
        self, ids: cv2.typing.MatLike, corners: Sequence[cv2.typing.MatLike]
    ) -> bool:
        """
        Update the transformation matrix from world to the robot frame.

        Params:
            ids: The ids of the aruco markers detected in an image.
            corners: The corresponding corner locations of the aruco markers in the
                image.

        Returns:
            True if tf_wr was updated successfully.
        """
        if self._tf_cw is None:
            return False

        robot_ids = np.array(list(self._robot_markers.keys()))
        idx = np.isin(ids, robot_ids)
        robot_points = []
        corners = np.array(corners)

        # PERF: This should be precomputed in init and then result can just be
        # indexed
        for id, corner in zip(ids[idx], corners[idx]):
            # Add all of the points detected on the robot in the robot frame
            marker_points_homo = np.column_stack(
                (self._marker_points, np.ones(self._marker_points.shape[0]))
            )
            points = (self._robot_markers[id] @ marker_points_homo.T).T[:, :-1]
            robot_points.append(points)

        if len(robot_points) < 1:
            print("short")
            return False

        robot_points = np.vstack(robot_points)
        success, r_vec, t_vec = cv2.solvePnP(
            robot_points,
            np.vstack(corners[idx]),
            self._mtx,
            self._dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            print("Failed")
            return False

        tf_cr = np.eye(4)
        R, _ = cv2.Rodrigues(r_vec)
        tf_cr[:3, :3] = R
        tf_cr[:3, 3] = t_vec.T

        tf_wc = np.linalg.inv(self._tf_cw)
        self._tf_wr = tf_wc @ tf_cr

        for id in ids[idx]:
            draw_axes(
                self._frame,
                0.15,
                tf_cr @ self._robot_markers[id],
                self._mtx,
                self._dist,
            )
        return True


def extract_pose_from_transform(tf: npt.NDArray) -> tuple[float, float, float]:
    """
    Extracts x, y, and heading (rotation around z-axis) from a 3D transformation matrix.

    Params:
        tf: The Homogeneous Transformation matrix.

    Returns:
        x: Translation in x-direction (m)
        y: Translation in y-direction (m)
        theta: Rotation about the z-axis (rad)
    """

    # Extract x and y translations
    x = tf[0, 3]
    y = tf[1, 3]

    # Extract heading (rotation about z-axis)
    theta = np.arctan2(tf[1, 0], tf[0, 0])

    return x, y, theta


def extract_pose_from_transform(tf: npt.NDArray) -> tuple[float, float, float]:
    """
    Extracts x, y, and heading (rotation around z-axis) from a 3D transformation matrix.

    Params:
        tf: The Homogeneous Transformation matrix.

    Returns:
        x: Translation in x-direction (m)
        y: Translation in y-direction (m)
        theta: Rotation about the z-axis (rad)
    """

    # Extract x and y translations
    x = tf[0, 3]
    y = tf[1, 3]

    # Extract heading (rotation about z-axis)
    theta = np.arctan2(tf[1, 0], tf[0, 0])

    return x, y, theta


def get_cam_params(file: str) -> tuple[np.ndarray, np.ndarray]:
    """Return mtx, dist camera parameters from yaml file"""
    with open(file, "r") as f:
        params = yaml.safe_load(f)

    mtx_shape = (
        params["camera_matrix"]["rows"],
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


def draw_axes(img, marker_size, tf, mtx, dist) -> None:
    tvec = tf[:3, 3].T
    rvec, _ = cv2.Rodrigues(tf[:3, :3])
    cv2.drawFrameAxes(img, mtx, dist, rvec, tvec, marker_size)


def vecs_to_tf(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Transform rvecs and tvecs to transformation matrix"""
    tf = np.eye(4)
    R, _ = cv2.Rodrigues(rvec)
    tf[:3, :3] = R
    tf[:3, 3] = tvec.T
    return tf


def tf_to_vecs(tf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Get rvecs and tvecs from transformation matrix"""
    rvec = cv2.Rodrigues(tf[:3, :3])
    tvec = tf[:3, 3].T
    return rvec, tvec


def vecs_to_tf(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Transform rvecs and tvecs to transformation matrix"""
    tf = np.eye(4)
    R, _ = cv2.Rodrigues(rvec)
    tf[:3, :3] = R
    tf[:3, 3] = tvec.T
    return tf


def tf_to_vecs(tf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Get rvecs and tvecs from transformation matrix"""
    rvec = cv2.Rodrigues(tf[:3, :3])
    tvec = tf[:3, 3].T
    return rvec, tvec


def main(args=None):
    # Example Usage
    dev = args.device

    cap = BufferlessVideoCapture(dev)
    cap = BufferlessVideoCapture(dev)

    mtx, dist = get_cam_params(args.cam_params)

    # Locate the one marker
    loc = Localisation(
        mtx, dist, ROBOT_MARKERS, marker_size=args.square, dict_type=args.size
    )

    while True:
        frame = cap.read()
        frame = cap.read()

        loc.add_img(frame)
        tf_cw = loc.tf_cw
        if tf_cw is not None:
            draw_axes(frame, args.square, tf_cw, mtx, dist)

            if loc.tf_wr is not None:
                tf_cr = tf_cw @ loc.tf_wr
                draw_axes(frame, args.square, tf_cr, mtx, dist)

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            exit(0)

        print(loc.tf_wr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("device")
    parser.add_argument("cam_params")
    parser.add_argument("--size", default=DEFAULT_SIZE)
    parser.add_argument("--square", type=float, default=0.12)
    main(parser.parse_args())
