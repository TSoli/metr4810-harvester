import math
import time

import motors
from drive import DifferentialDrive
from machine import Pin
from motors import DCMotor

RPM_TO_RAD_S = 2 * math.pi / 60


def main():
    onboard = Pin(25, Pin.OUT, value=1)
    max_rpm = 100
    max_motor_speed = max_rpm * RPM_TO_RAD_S
    left_drive_motor = DCMotor(
        motors.LEFT_MOTOR_PWM_PIN, motors.LEFT_MOTOR_DIR_PIN, max_motor_speed
    )
    right_drive_motor = DCMotor(
        motors.RIGHT_MOTOR_PWM_PIN, motors.RIGHT_MOTOR_DIR_PIN, max_motor_speed
    )
    drive_radius = 0.0837  # distance from centre to tracks
    wheel_radius = 0.0396
    drive = DifferentialDrive(
        left_drive_motor, right_drive_motor, drive_radius, wheel_radius
    )

    time.sleep(1)
    # negative speed makes drive motors go forward
    while True:
        # drive forward
        drive.drive(0.5 * drive.v_max, 0)
        time.sleep(2)
        # turn around left
        drive.drive(0, math.pi / 4)
        time.sleep(2)


if __name__ == "__main__":
    main()
