
import requests
from config import Config

class MessageTypes:
    DRIVE = "drive"
    SCOOP="scoop"
    LED="led"
    CONTAINER="container"

def send_command(data):
    """
    Send a command to the Pico W device.
    
    Args:
        data (dict): The command data to be sent.
    
    Returns:
        bool: True if the request was successful, False otherwise.
    """
        
    response = requests.post(Config.PICO_W_ADDRESS, json = data)
    print(f"Response Code: {response.status_code}")
    print(f"Response: {response.text}")
    return response.ok

def send_drive_request(v, w):
    """
    Send a drive command to control the robot's movement.
    
    Args:
        v (float): Linear velocity.
        w (float): Angular velocity.
    
    Returns:
        bool: True if the request was successful, False otherwise.
    """
    request = {
        "type": MessageTypes.DRIVE,
        "v": v,
        "w": w
    }
    return send_command(request)

def send_scoop_request(direction):
    """
    Send a scoop command to control the robot's scoop mechanism.
    
    Args:
        direction (str): The direction of the scoop movement.
    
    Returns:
        bool: True if the request was successful, False otherwise.
    """
      
    request = {
        "type": MessageTypes.SCOOP,
        "direction": direction
    }
    return send_command(request)

def send_led_request(on: bool):
    """
    Send a command to control the LED.
    
    Args:
        on (bool): True to turn the LED on, False to turn it off.
    
    Returns:
        bool: True if the request was successful, False otherwise.
    """
    request = {
        "type": MessageTypes.LED,
        "on": 1 if on else 0
    }
    return send_command(request)


def send_container_request(open: bool):
    """
    Send a command to control the container mechanism.
    
    Args:
        open (bool): True to open the container, False to close it.
    
    Returns:
        bool: True if the request was successful, False otherwise.
    """
      
    request = {
        "type": MessageTypes.CONTAINER,
        "open": open
    }
    return send_command(request)
