import sys

from motors import DCMotor


class DifferentialDrive:
    """Models a differential drive robot"""

    def __init__(
        self,
        left_motor: DCMotor,
        right_motor: DCMotor,
        max_speed: float,
        radius: float,
    ) -> None:
        """
        Params:
            left_motor: The left motor of the robot.
            right_motor: The right motor of the robot.
            max_speed: The max speed of wheels in m/s.
            drive_radius: The distance from the centre of the robot to the drive motors.
        """
        self._left_motor = left_motor
        self._right_motor = right_motor
        self._max_speed = max_speed
        self._drive_radius = radius

    def drive(self, v: float, omega: float) -> None:
        """
        Set the drive velocities for the motors.

        Params:
            v: The linear velocity of the robot in m/s.
            omega: The angular velocity of the robot in rad/s (assuming the vector)
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
        vl_norm = vl / self._max_speed
        vr_norm = vr / self._max_speed
        self._left_motor.speed = vl_norm
        self._right_motor = vr_norm

        if not (-1 < vl_norm < 1):
            print(f"vl was clipped: vl={vl_norm}", file=sys.stderr)

        if not (-1 < vr_norm < 1):
            print(f"vr was clipped: vl={vr_norm}", file=sys.stderr)
