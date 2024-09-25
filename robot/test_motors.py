import time

import motors
from machine import Pin


def main():
    onboard = Pin(25, Pin.OUT, value=1)
    left_drive_motor = motors.DCMotor(
        motors.LEFT_MOTOR_PWM_PIN, motors.LEFT_MOTOR_DIR_PIN
    )
    right_drive_motor = motors.DCMotor(
        motors.RIGHT_MOTOR_PWM_PIN, motors.RIGHT_MOTOR_DIR_PIN
    )

    time.sleep(1)
    # negative speed makes drive motors go forward
    while True:
        # drive forward
        left_drive_motor.speed = -0.5
        right_drive_motor.speed = -0.5
        time.sleep(10)
        # turn clockwise
        right_drive_motor.speed = 0.5
        time.sleep(2)
