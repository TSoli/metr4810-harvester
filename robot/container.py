import math

from motors import Servo


class Container:
    """Open and close the container"""

    def __init__(self, servo: Servo):
        """
        Params:
            servo: the servo that controls the container.
        """
        self._servo = servo

    def open(self) -> None:
        """Open the container"""
        self._servo.angle = math.pi

    def close(self) -> None:
        """Close the container"""
        self._servo.angle = 0
