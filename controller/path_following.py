import time

import numpy as np
from logger import logger
from numpy.typing import NDArray


class PurePursuitController:
    """A controller that implements the Pure Pursuit algorithm."""

    def __init__(self, look_ahead: float, avg_speed: float, tol: float = 15e-3) -> None:
        """
        Params:
            look_ahead: The distance to look ahead for a point to head towards.
            avg_speed: The average speed for the controller.
            tol: A point is considered reached if within this tolerance.
        """
        self._look_ahead = look_ahead
        self._avg_speed = avg_speed
        self._tol = tol
        self._path = np.empty([])

    @property
    def path(self) -> NDArray[np.float64]:
        """The path to follow with nodes (x, y, theta)"""
        return self._path

    @path.setter
    def path(self, path: NDArray[np.float64]):
        """Set the path to follow with nodes (x, y, theta)"""
        self._path = path

    def get_control_action(self, pose: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Get the required linear and angular velocity to follow the current path
        from the current position.

        Params:
            pose: The current pose of the robot (x, y, theta)

        Returns:
            velocity (m/s), angular velocity (rad/s). If both are 0, the end of
            the path has been reached.
        """
        distance, idx = self._find_closest_point(pose)
        if idx == len(self._path) - 1 and distance < self._tol:
            return np.array([0, 0])

        look_ahead_point = self._find_goal_point(idx)
        return self._find_control_action(pose, look_ahead_point)

    @property
    def avg_speed(self) -> float:
        return self._avg_speed

    @avg_speed.setter
    def avg_speed(self, speed: float) -> None:
        self._avg_speed = speed

    def _find_closest_point(self, pose: NDArray[np.float64]) -> tuple[np.float64, int]:
        """
        Find the closest point on the path to the robot.

        Params:
            pose: The current pose (x, y, theta)

        Returns:
            distance (m), index - The distance to the closest point and its
            index in the path.
        """
        # Find the closest point on the path
        distances = np.linalg.norm(self._path[:, :2] - pose[:2], axis=1)
        closest_index = np.argmin(distances)
        return distances[closest_index], closest_index  # type: ignore

    def _find_goal_point(self, idx: int) -> NDArray[np.float64]:
        """
        Find the point on the path to head towards.

        Params:
            idx: The index on the path to start searching from.

        Returns:
            The goal point.
        """
        from_point = self._path[idx]
        goal_point = self._path[idx]
        for pt in self._path[idx:]:
            if np.linalg.norm(pt[:2] - from_point[:2]) > self._look_ahead:
                break
            goal_point = pt

        logger.info(f"Goal point: {goal_point}")
        return goal_point

    def _find_control_action(
        self, pose: NDArray[np.float64], goal_point: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Find the control action to follow the path.

        Params:
            pose: The current pose (x, y, theta)
            goal_point: The point to head towards (x, y, theta)

        Returns:
            velocity (m/s), angular velocity (rad/s)
        """
        # transform from world to base
        tf_wb = np.array(
            [
                [np.cos(pose[2]), -np.sin(pose[2]), pose[0]],
                [np.sin(pose[2]), np.cos(pose[2]), pose[1]],
                [0, 0, 1],
            ]
        )
        tf_bw = np.linalg.inv(tf_wb)
        goal_point_b = tf_bw @ np.array([goal_point[0], goal_point[1], 1])
        curvature = -2 * goal_point_b[0] / np.linalg.norm(goal_point_b[:2]) ** 2
        distance = np.linalg.norm(goal_point_b[:2])
        velocity = self._avg_speed

        if distance < 3 * self._tol:
            # TODO: Slow down when close?
            velocity *= 0.4

        if goal_point_b[1] < 0:
            velocity *= -1

        omega = velocity * curvature
        return np.array([velocity, omega])


class HeadingController:
    """A PI controller for the robot heading"""

    def __init__(self, kp: float, ki: float, tol: float = np.deg2rad(5)) -> None:
        """
        Params:
            tol: Tolerance in rad for reaching the desired heading.
        """
        self._tol = tol
        self._kp = kp
        self._ki = ki
        self._integral = 0
        self._last_time = None

    def reset(self) -> None:
        """Reset the integral"""
        self._integral = 0
        self._last_time = None

    def get_control_action(self, heading: float, goal: float) -> float:
        """
        Get an angular velocity control action to reach the goal heading.

        Params:
            heading: The current heading of the robot.
            goal: The goal heading.

        Returns:
            angular velocity of robot (rad/s).
        """
        error = goal - heading
        if abs(error) > np.pi:
            err_sgn = np.sign(error)
            error = -err_sgn * ((2 * np.pi) - abs(error))

        if abs(error) < self._tol:
            return 0

        if self._last_time is not None:
            dt = time.time() - self._last_time
            self._integral += dt * error

        self._last_time = time.time()
        return self._kp * error + self._ki * self._integral
