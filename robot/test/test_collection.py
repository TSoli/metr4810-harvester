import math
import time

import motors
from drive import DifferentialDrive
from machine import Pin
from motors import DCMotor

RPM_TO_RAD_S = 2 * math.pi / 60


def lift_motor(scoop_motor: motors.Servo, vibration_motor: motors.DCMotor) -> None:
    scoop_motor.angle = scoop_motor.max_angle
    time.sleep(0.5)
    for i in range(10):
        scoop_motor.angle = scoop_motor.max_angle - (i * math.pi / 36)
        time.sleep(0.1)

    vibration_motor.speed = 0
    time.sleep(0.2)
    scoop_motor.angle = scoop_motor.max_angle - math.pi


def drop_motor(scoop_motor: motors.Servo, vibration_motor: motors.DCMotor) -> None:
    scoop_motor.angle = scoop_motor.max_angle - (math.pi / 4)
    time.sleep(0.2)
    vibration_motor.speed = 0.6 * vibration_motor.max_speed
    for i in range(10):
        scoop_motor.angle = scoop_motor.max_angle - (math.pi / 4) + (math.pi * i / 36)
        time.sleep(0.1)

    scoop_motor.angle = scoop_motor.max_angle


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
    scoop_motor = motors.Servo(motors.ARM_MOTOR_PWM_PIN)
    max_vib_speed = 100 * RPM_TO_RAD_S
    vibration_motor = motors.DCMotor(10, 11, max_vib_speed, freq=50_000)
    drive_radius = 0.0837  # distance from centre to tracks
    wheel_radius = 0.0396
    drive = DifferentialDrive(
        left_drive_motor, right_drive_motor, drive_radius, wheel_radius
    )

    time.sleep(1)
    scoop_motor.angle = scoop_motor.max_angle
    vibration_motor.speed = 0.6 * vibration_motor.max_speed
    # extra_motor.speed = 0.5 * extra_motor.max_speed
    time.sleep(1)
    # drive.drive(0.5 * drive.v_max, 0)
    while True:
        drive.drive(0.5 * drive.v_max, 0)
        time.sleep(2)
        drive.drive(0, 0)
        lift_motor(scoop_motor, vibration_motor)
        time.sleep(1)
        drop_motor(scoop_motor, vibration_motor)


if __name__ == "__main__":
    main()
