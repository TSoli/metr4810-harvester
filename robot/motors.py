import math

from machine import PWM, Pin

U16_MAX = 65535
LEFT_MOTOR_PWM_PIN = 14
LEFT_MOTOR_DIR_PIN = 15
RIGHT_MOTOR_PWM_PIN = 12
RIGHT_MOTOR_DIR_PIN = 13
ARM_MOTOR_PWM_PIN = 2  # Use MISC servo pin due to clearance on PCB


class DCMotor:
    """Represents a DC motor"""

    def __init__(self, pwm: int, dir: int, freq: int = 1000) -> None:
        """
        Params:
            pwm: The PWM GPIO pin number.
            dir: The pin number that controls the motor direction.
            freq: The PWM frequency, defaults to 1000.
        """
        self._pwm = PWM(Pin(pwm, Pin.OUT), freq=freq, duty_u16=0)
        self._dir = Pin(dir, Pin.OUT, value=0)
        self._speed = 0

    @property
    def speed(self) -> float:
        """Motor speed as a value between -1 and 1"""
        return self._speed

    @speed.setter
    def speed(self, val: float) -> None:
        """Set the motor speed to a value between -1 and 1"""
        self._speed = max(min(val, 1), -1)

        if self._speed < 0:
            self._dir.value(0)
        else:
            self._dir.value(1)

        self._pwm.duty_u16(round(abs(self._speed * U16_MAX)))


class Servo:
    """Represents a servo motor"""

    def __init__(
        self,
        pin: int,
        freq: int = 200,
        max_angle: float = 3 * math.pi / 2,
        min_pulse: int = 500_000,
        max_pulse: int = 2_500_000,
    ) -> None:
        """
        Params:
            pin: The PWM control GPIO pin number.
            freq: The PWM freq in Hz, defaults to 200.
            max_angle: The maximum angle of the servo in radians.
            min_pulse: The minimum pulse length (in nanoseconds) corresponding
                to an angle of 0, defaults to 500k.
            max_pulse: The maximum pulse length (in nanoseconds) corresponding
                to max_angle, defaults to 2.5 million.
        """
        self._pin = PWM(Pin(pin, Pin.OUT), freq=freq, duty_ns=0)
        self._angle = 0
        self._max_angle = max_angle
        self._min_pulse = min_pulse
        self._max_pulse = max_pulse
        self._pulse_delta = self._max_pulse - self._min_pulse

    @property
    def angle(self) -> float:
        """The angle of the servo motor in radians"""
        return self._angle

    @angle.setter
    def angle(self, val: float) -> None:
        """Set the servo motor angle in radians"""
        self._angle = max(min(val, self.max_angle), 0)
        angle_ratio = self._angle / self.max_angle
        self._pin.duty_ns(round(self._min_pulse + angle_ratio * self._pulse_delta))

    @property
    def max_angle(self) -> float:
        """The maximum angle of the servo motor in radians"""
        return self._max_angle
