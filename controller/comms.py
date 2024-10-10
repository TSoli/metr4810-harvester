import json
import typing

import requests
from logger import logger


class MessageTypes:
    """Message types that can be sent to the robot"""

    DRIVE = "drive"
    SCOOP = "scoop"
    LED = "led"
    CONTAINER = "container"


class Comms:
    """Handle communications with the robot"""

    def __init__(self, ip: str) -> None:
        """
        Params:
            ip: The ip address to connect to for communications.
        """
        self._session = requests.Session()
        self._ip = ip

    def send_drive_request(self, v: float, w: float) -> bool:
        """
        Send a drive command to control the robot's movement.

        Args:
            v (float): Linear velocity.
            w (float): Angular velocity.

        Returns:
            bool: True if the request was successful, False otherwise.
        """
        request = {"type": MessageTypes.DRIVE, "v": v, "w": w}
        return self._send_command(request)

    def send_scoop_request(self, direction: str) -> bool:
        """
        Send a scoop command to control the robot's scoop mechanism.

        Args:
            direction (str): The direction of the scoop movement.

        Returns:
            bool: True if the request was successful, False otherwise.
        """

        request = {"type": MessageTypes.SCOOP, "direction": direction}
        return self._send_command(request)

    def send_led_request(self, on: bool) -> bool:
        """
        Send a command to control the LED.

        Args:
            on (bool): True to turn the LED on, False to turn it off.

        Returns:
            bool: True if the request was successful, False otherwise.
        """
        request = {"type": MessageTypes.LED, "on": 1 if on else 0}
        return self._send_command(request)

    def send_container_request(self, open: bool) -> bool:
        """
        Send a command to control the container mechanism.

        Args:
            open (bool): True to open the container, False to close it.

        Returns:
            bool: True if the request was successful, False otherwise.
        """

        request = {"type": MessageTypes.CONTAINER, "open": open}
        return self._send_command(request)

    def _send_command(self, data: dict[str, typing.Any]) -> bool:
        """Send a command to the Pico W device.

        Args:
            data (dict): The command data to be sent. Must have field type
            for the message type.

        Returns:
            bool: True if the request was successful, False otherwise.
        """
        if "type" not in data:
            logger.warn(f"Sending command without type:\n{data}")

        logger.info(f"Sending:\n{data}")
        response = self._session.post(self._ip, data=json.dumps(data))
        logger.info(f"Response:\nHeaders:\n{response.headers}\nText:{response.text}")
        return response.ok
