import math
import time

from motors import DCMotor, Servo


class Scoop:
    """Controls the scoop"""

    def __init__(self, servo: Servo, vib_motor: DCMotor) -> None:
        self._servo = servo
        self._vib_motor = vib_motor

    def up(self) -> None:
        """Lift the scoop"""
        for i in range(100):
            self._servo.angle = self._servo.max_angle - (i * math.pi / 360)
            time.sleep(0.01)

        self._vib_motor.speed = 0
        time.sleep(0.1)
        self._servo.angle = self._servo.max_angle - math.pi

    def down(self) -> None:
        """Lower the scoop"""
        self._servo.angle = self._servo.max_angle - (math.pi / 4)
        time.sleep(0.2)
        self._vib_motor.speed = 0.6 * self._vib_motor.max_speed
        for i in range(100):
            self._servo.angle = (
                self._servo.max_angle - (100 * math.pi / 360) + (math.pi * i / 360)
            )
            time.sleep(0.01)

        self._servo.angle = self._servo.max_angle
