import math
import time

import motors
from drive import DifferentialDrive
from machine import Pin
from motors import DCMotor

RPM_TO_RAD_S = 2 * math.pi / 60


def main():
    scoop_motor = motors.Servo(motors.ARM_MOTOR_PWM_PIN)

    time.sleep(1)
    scoop_motor.angle = scoop_motor.max_angle - math.pi
    while True:
        time.sleep(5)
        scoop_motor.angle = scoop_motor.max_angle
        time.sleep(5)
        scoop_motor.angle = scoop_motor.max_angle - math.pi


if __name__ == "__main__":
    main()
