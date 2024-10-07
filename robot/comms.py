import json
import socket
import time

import network
from logger import logger

WIFI_FILENAME = "wifi.json"


def connect() -> str:
    """
    Connect to the Wifi network with details in WIFI_FILENAME json file

    Returns:
        The ip address of the device.
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

    return wlan.ifconfig()[0]


def open_socket(ip: str, port: int) -> socket.socket:
    """
    Open a socket at ip:port.

    Returns:
        The socket.
    """
    addr = (ip, port)
    connection = socket.socket()
    connection.bind(addr)
    # Only one device should connect - the controller
    connection.listen(1)
    return connection
