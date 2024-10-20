import math
import time

import motors
from motors import DCMotor, Servo
from scoop import Scoop

RPM_TO_RAD_S = 2 * math.pi / 60


def main():
    scoop_motor = Servo(motors.SCOOP_MOTOR_PWM_PIN)
    vib_motor = DCMotor(
        motors.VIB_MOTOR_PWM_PIN, motors.VIB_MOTOR_DIR_PIN, 6e3 * RPM_TO_RAD_S, 50_000
    )
    scoop = Scoop(scoop_motor, vib_motor)

    time.sleep(1)
    scoop.up()
    while True:
        time.sleep(5)
        scoop.down()
        time.sleep(5)
        scoop.up()


if __name__ == "__main__":
    main()
