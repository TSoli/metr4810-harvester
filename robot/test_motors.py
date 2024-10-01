import time

import motors
from logger import logger
from machine import Pin


def main():
    logger.info("Starting")
    time.sleep(1)
    onboard = Pin(25, Pin.OUT, value=1)
    logger.debug("LED on")
    left_drive_motor = motors.DCMotor(
        motors.LEFT_MOTOR_PWM_PIN, motors.LEFT_MOTOR_DIR_PIN
    )
    right_drive_motor = motors.DCMotor(
        motors.RIGHT_MOTOR_PWM_PIN, motors.RIGHT_MOTOR_DIR_PIN
    )
    # arm_motor = motors.Servo(motors.ARM_MOTOR_PWM_PIN)
    #
    # arm_angle = 0
    # time.sleep(1)
    # # negative speed makes drive motors go forward
    # while True:
    #     # drive forward
    #     onboard.value(0)
    #     left_drive_motor.speed = -0.5
    #     right_drive_motor.speed = -0.5
    #     arm_motor.angle = arm_angle
    #     time.sleep(2)
    #     onboard.value(1)
    #     arm_motor.angle = arm_motor.max_angle
    #     # turn clockwise
    #     right_drive_motor.speed = 0.5
    #     time.sleep(2)


if __name__ == "__main__":
    main()
