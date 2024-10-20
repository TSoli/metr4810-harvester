import json
import socket
import time
from queue import Queue

import network
from logger import logger

WIFI_FILENAME = "wifi.json"


class Comms:
    """Handles communications for the robot."""

    def __init__(self, commands: Queue, port: int = 80) -> None:
        """
        Params:
            commands: A queue that stores commands for the robot to follow.
            ip: The ip to connect with.
            port: The port to set up a socket on.
        """
        self._commands = commands
        self._ip = "0.0.0.0"
        self._port = port
        self._socket = None
        self._wlan = None

    def connect(self) -> None:
        """
        Connect to the Wifi network with details in WIFI_FILENAME json file

        Side-effects:
            Sets the ip.
        """
        wlan = network.WLAN(network.STA_IF)
        wlan.active(True)
        with open("wifi.json", "r") as f:
            data = json.load(f)

        ssid = data["ssid"]
        password = data["password"]
        # ip_info = wlan.ifconfig()
        # ip_config = (self._ip, ip_info[1], ip_info[2], ip_info[3])
        # wlan.ifconfig(ip_config)
        while True:
            try:
                wlan.connect(ssid, password)

                while not wlan.isconnected():
                    time.sleep(0.1)

                break
            except Exception as e:
                pass

        self._ip = wlan.ifconfig()[0]
        self._wlan = wlan
        # logger.info(f"Connected to {self._ip}")

    def run(self) -> None:
        """Start accepting messages"""
        self._open_socket()

        while True:
            if not self._wlan.isconnected():
                self.connect()
            try:
                data = self._socket.recv(1024).decode()
            except Exception as e:
                continue

            if not data:
                continue

            try:
                command = json.loads(data)
            except Exception as e:
                continue
            self._queue_command(command)

    def _open_socket(self) -> None:
        """
        Open a socket at ip:port.

        Side-effects:
            Opens a socket.
        """
        addr = (self._ip, self._port)
        connection = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        connection.bind(addr)
        connection.setblocking(False)
        self._socket = connection

    def _queue_command(self, command: dict) -> None:
        """
        Send command to queue

        Will block until the queue is empty.
        """
        while not self._commands.push(command):
            pass
