from logger import logger
from motors import DCMotor


class DifferentialDrive:
    """Models a differential drive robot"""

    def __init__(
        self,
        left_motor: DCMotor,
        right_motor: DCMotor,
        drive_radius: float,
        wheel_radius: float,
    ) -> None:
        """
        Params:
            left_motor: The left motor of the robot.
            right_motor: The right motor of the robot.
            drive_radius: The distance from the centre of the robot to the drive motors.
            wheel_radius: The radius of the wheels.
        """
        self._left_motor = left_motor
        self._right_motor = right_motor
        self._drive_radius = drive_radius
        self._wheel_radius = wheel_radius
        # left and right motors are assumed to have the same max speed.
        self._v_max = self._wheel_radius * self._left_motor.max_speed
        self._omega_max = self._v_max * self._drive_radius

    @property
    def v_max(self) -> float:
        """
        The maximum speed of the drive system in m/s.

        Note this is only achievable if moving straight.
        """
        return self._v_max

    @property
    def omega_max(self) -> float:
        """
        The maximum angular velocity of the drive system in m/s.

        Note this is only achievable if rotating on the spot.
        """
        return self._omega_max

    def drive(self, v: float, omega: float) -> None:
        """
        Set the drive velocities for the motors.

        Params:
            v: The linear velocity of the system in m/s.
            omega: The angular velocity of the system in rad/s (assuming the vector)
                points up out of the robot).

        Returns:
            True if the motor velocities could be set without clipping. False,
            otherwise.

        Side-effects:
            If the desired wheel velocities could not be achieved a message is printed
            to stderr.
        """
        vl = v - self._drive_radius * omega
        vr = v + self._drive_radius * omega
        l_speed = vl / self._wheel_radius
        r_speed = vr / self._wheel_radius

        self._left_motor.speed = l_speed
        self._right_motor.speed = r_speed
        if not (abs(l_speed) <= self._left_motor.max_speed):
            logger.warn(
                f"Left motor speed was clipped from {l_speed} to {self._left_motor.speed}"
            )

        if not abs(r_speed) <= self._right_motor.max_speed:
            logger.warn(
                f"Right motor speed was clipped from {r_speed} to {self._right_motor.speed}"
            )
