import math
import time

import motors
from container import Container
from drive import DifferentialDrive
from machine import Pin
from motors import DCMotor, Servo
from scoop import Scoop

RPM_TO_RAD_S = 2 * math.pi / 60


RPM_TO_RAD_S = 2 * math.pi / 60
DRIVE_RADIUS = 0.0837  # distance from centre to tracks in m
WHEEL_RADIUS = 0.0396  # radius of wheel in m

Led = Pin


def setup_actuators() -> tuple[DifferentialDrive, Scoop, Container, Led]:
    """Setup the motors"""
    left_drive_motor = DCMotor(
        motors.LEFT_MOTOR_PWM_PIN,
        motors.LEFT_MOTOR_DIR_PIN,
        motors.DRIVE_MAX_RPM * RPM_TO_RAD_S,
    )
    right_drive_motor = DCMotor(
        motors.RIGHT_MOTOR_PWM_PIN,
        motors.RIGHT_MOTOR_DIR_PIN,
        motors.DRIVE_MAX_RPM * RPM_TO_RAD_S,
    )
    scoop_motor = Servo(motors.SCOOP_MOTOR_PWM_PIN)
    vib_motor = DCMotor(
        motors.VIB_MOTOR_PWM_PIN,
        motors.VIB_MOTOR_DIR_PIN,
        motors.VIB_MAX_RPM * RPM_TO_RAD_S,
        freq=50_000,
    )
    container_servo = Servo(motors.CONTAINER_MOTOR_PWM_PIN)
    container = Container(container_servo)
    drive = DifferentialDrive(
        left_drive_motor, right_drive_motor, DRIVE_RADIUS, WHEEL_RADIUS
    )
    scoop = Scoop(scoop_motor, vib_motor)
    led = Pin(25, Pin.OUT)
    return drive, scoop, container, led


def main():
    drive, scoop, container, led = setup_actuators()
    scoop.up()
    time.sleep(1)
    drive.drive(0.5 * drive.v_max, 0)
    time.sleep(2)
    drive.drive(0, 0)
    scoop.down()
    time.sleep(1)


if __name__ == "__main__":
    main()
