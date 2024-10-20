import json
import socket
import typing

from logger import logger


class MessageTypes:
    """Message types that can be sent to the robot"""

    DRIVE = "drive"
    SCOOP = "scoop"
    LED = "led"
    CONTAINER = "container"


class Comms:
    """Handle communications with the robot"""

    def __init__(self, ip: str, port: int = 80) -> None:
        """
        Params:
            ip: The ip address to connect to for communications.
        """
        self._ip = ip
        self._port = port
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def __del__(self) -> None:
        """Clean up"""
        self._socket.close()

    def send_drive_request(self, v: float, w: float) -> None:
        """
        Send a drive command to control the robot's movement.

        Args:
            v (float): Linear velocity.
            w (float): Angular velocity.

        Returns:
            bool: True if the request was successful, False otherwise.
        """
        request = {"type": MessageTypes.DRIVE, "v": float(v), "w": float(w)}
        self._send_command(request)

    def send_scoop_request(self, up: bool) -> None:
        """
        Send a scoop command to control the robot's scoop mechanism.

        Args:
            up: True to lift scoop or False to lower.

        Returns:
            bool: True if the request was successful, False otherwise.
        """

        request = {"type": MessageTypes.SCOOP, "up": up}
        self._send_command(request)

    def send_led_request(self, on: bool) -> None:
        """
        Send a command to control the LED.

        Args:
            on (bool): True to turn the LED on, False to turn it off.

        Returns:
            bool: True if the request was successful, False otherwise.
        """
        request = {"type": MessageTypes.LED, "on": on}
        self._send_command(request)

    def send_container_request(self, open: bool) -> None:
        """
        Send a command to control the container mechanism.

        Args:
            open (bool): True to open the container, False to close it.

        Returns:
            bool: True if the request was successful, False otherwise.
        """

        request = {"type": MessageTypes.CONTAINER, "open": open}
        self._send_command(request)

    def _send_command(self, data: dict[str, typing.Any]) -> None:
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
        msg = json.dumps(data)
        self._socket.sendto(msg.encode(), (self._ip, self._port))
