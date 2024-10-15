import json
import math

from comms import Comms
from pynput import keyboard

DRIVE_MAX_RPM = 100
DRIVE_MAX_RAD_S = DRIVE_MAX_RPM * 2 * math.pi / 60

DRIVE_RADIUS = 0.0837  # distance from centre to tracks in m
WHEEL_RADIUS = 0.0396  # radius of wheel in m


class Controller:
    def __init__(self, max_v: float, max_w: float, comms: Comms):
        """
        Params:
            max_v: Max velocity in m/s.
        """
        self._max_v = max_v
        self._max_w = max_w
        self._comms = comms
        self._v = 0.0
        self._w = 0.0
        self._scoop = False
        self._container = False
        self._led = False

    def on_press(self, key) -> None:
        try:
            char = key.char
        except AttributeError:
            return

        if char == "w":
            self._v = float(min(self._v + 0.01, self._max_v))
            print(f"v: {self._v:.2f} w: {self._w:.2f}")
            self._comms.send_drive_request(self._v, self._w)
        elif char == "x":
            self._v = float(max(self._v - 0.01, -self._max_v))
            print(f"v: {self._v:.2f} w: {self._w:.2f}")
            self._comms.send_drive_request(self._v, self._w)
        elif char == "s":
            # stop
            self._v = 0.0
            self._w = 0.0
            print("Stop!")
            self._comms.send_drive_request(self._v, self._w)
        elif char == "a":
            self._w = float(min(self._w + math.pi / 16, self._max_w))
            print(f"v: {self._v:.2f} w: {self._w:.2f}")
            self._comms.send_drive_request(self._v, self._w)
        elif char == "d":
            self._w = float(max(self._w - math.pi / 16, -self._max_w))
            print(f"v: {self._v:.2f} w: {self._w:.2f}")
            self._comms.send_drive_request(self._v, self._w)
        elif char == "q":
            self._scoop = True
            print(f"Scoop up")
            self._comms.send_scoop_request(self._scoop)
        elif char == "z":
            self._scoop = False
            print(f"Scoop down")
            self._comms.send_scoop_request(self._scoop)
        elif char == "e":
            self._container = True
            print(f"Open container")
            self._comms.send_container_request(self._container)
        elif char == "c":
            self._container = False
            print(f"Close contrainer")
            self._comms.send_container_request(self._container)
        elif char == "r":
            self._led = True
            print("LED on")
            self._comms.send_led_request(self._led)
        elif char == "f":
            self._led = False
            print("LED off")
            self._comms.send_led_request(self._led)


def main():
    with open("comms_config.json", "r") as f:
        data = json.load(f)

    ip = data["ip"]
    comms = Comms(ip)
    max_v = DRIVE_MAX_RAD_S * WHEEL_RADIUS
    max_w = max_v / DRIVE_RADIUS
    controller = Controller(max_v, max_w, comms)

    with keyboard.Listener(on_press=controller.on_press) as listener:
        listener.join()


if __name__ == "__main__":
    main()
