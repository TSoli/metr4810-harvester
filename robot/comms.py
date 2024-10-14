import json
import socket
import time
from queue import Queue

import network
from logger import logger

WIFI_FILENAME = "wifi.json"


class Comms:
    """Handles communications for the robot."""

    def __init__(self, commands: Queue, ip: str = "0.0.0.0", port: int = 80) -> None:
        """
        Params:
            commands: A queue that stores commands for the robot to follow.
            ip: The ip to connect with.
            port: The port to set up a socket on.
        """
        self._commands = commands
        self._ip = ip
        self._port = port
        self._socket = None

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
        while True:
            try:
                wlan.connect(ssid, password)

                logger.info("Connecting...")
                while not wlan.isconnected():
                    time.sleep(0.1)

                break
            except Exception as e:
                logger.info("Connection failed!")
                logger.info("Retrying")

        self._ip = wlan.ifconfig()[0]
        logger.info(f"Connected at {self._ip}")

    def run(self) -> None:
        """Start accepting messages"""
        self._open_socket()

        # TODO: Handle sending logs?
        while True:
            cl, addr = self._socket.accept()  # type: ignore
            logger.info(f"Client connected from: {addr}")
            while True:
                headers = cl.recv(1024).decode()
                if not headers:
                    cl.close()
                    logger.info(f"Closing connection to {addr}")
                    break

                content_length = int(
                    headers.split("Content-Length: ")[1].split("\r\n")[0]
                )

                if "Content-Type" in headers:
                    content_type = headers.split("Content-Type: ")[1].split("\r\n")[0]
                else:
                    content_type = None

                body = cl.recv(content_length).decode()
                # headers, body = request.split("\r\n\r\n", 1)

                # TODO: Update message? Do we even need a body/text?
                message = "Hello!"
                # Prepare response
                headers = f"HTTP/1.0 200 OK\r\nContent-type: text/plain\r\nContent-Length: {len(message)}\r\nConnection: keep-alive\r\n\r\n"
                cl.send(headers.encode())
                cl.send(message.encode())

                if content_type == "application/json":
                    command = json.loads(body)
                    self._queue_command(command)
                else:
                    logger.debug("Unknown message type")

    def _open_socket(self) -> None:
        """
        Open a socket at ip:port.

        Side-effects:
            Opens a socket (setting the socket property)
        """
        addr = (self._ip, self._port)
        connection = socket.socket()
        connection.bind(addr)
        # Only one device should connect - the controller
        connection.listen(1)
        self._socket = connection
        self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def _queue_command(self, command: dict) -> None:
        """
        Send command to queue

        Will block until the queue is empty.
        """
        start = time.time()
        while not self._commands.push(command):
            pass

        logger.debug(f"Took {(start - time.time()) * 1000}ms to send command to queue")
