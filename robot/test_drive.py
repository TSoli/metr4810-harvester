import math
import time

import motors
from drive import DifferentialDrive
from machine import Pin
from motors import DCMotor


def main():
    onboard = Pin(25, Pin.OUT, value=1)
    left_drive_motor = DCMotor(motors.LEFT_MOTOR_PWM_PIN, motors.LEFT_MOTOR_DIR_PIN)
    right_drive_motor = DCMotor(motors.RIGHT_MOTOR_PWM_PIN, motors.RIGHT_MOTOR_DIR_PIN)
    max_speed = 0.1  # m/s
    radius = 0.095  # distance from centre to tracks
    drive = DifferentialDrive(left_drive_motor, right_drive_motor, max_speed, radius)

    time.sleep(1)
    # negative speed makes drive motors go forward
    while True:
        # drive forward
        drive.drive(0.05, 0)
        time.sleep(2)
        # turn around
        drive.drive(0, math.pi / 2)
        time.sleep(2)


if __name__ == "__main__":
    main()
