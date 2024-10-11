import json
import os
import sys

import requests

parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_directory)

from comms import Comms


def main():
    with open(os.path.join("..", "comms_config.json")) as f:
        data = json.load(f)

    ip = data["ip"]
    comms = Comms(ip)
    comms.send_led_request(True)
    comms.send_drive_request(1, 1)
    comms.send_scoop_request("up")
    comms.send_container_request(False)


if __name__ == "__main__":
    main()
