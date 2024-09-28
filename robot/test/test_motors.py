import math
import time

import motors
# from logger import logger
from machine import Pin

RPM_TO_RAD_PER_S = 2 * math.pi / 60


def main():
    time.sleep(1)
    onboard = Pin(25, Pin.OUT, value=1)
    max_rpm = 100
    max_speed = max_rpm * RPM_TO_RAD_PER_S
    left_drive_motor = motors.DCMotor(
        motors.LEFT_MOTOR_PWM_PIN, motors.LEFT_MOTOR_DIR_PIN, max_speed
    )
    right_drive_motor = motors.DCMotor(
        motors.RIGHT_MOTOR_PWM_PIN, motors.RIGHT_MOTOR_DIR_PIN, max_speed
    )
    arm_motor = motors.Servo(motors.ARM_MOTOR_PWM_PIN)
    arm_angle = 0
    time.sleep(1)
    while True:
        # drive forward
        onboard.value(0)
        left_drive_motor.speed = 0.5 * left_drive_motor.max_speed
        right_drive_motor.speed = 0.5 * right_drive_motor.max_speed
        arm_motor.angle = arm_angle
        right_drive_motor.speed = -0.5 * right_drive_motor.max_speed
        time.sleep(2)


if __name__ == "__main__":
    main()
