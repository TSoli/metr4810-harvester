import _thread
import math
from queue import Queue

import motors
from comms import Comms
from container import Container
from drive import DifferentialDrive
from logger import logger
from machine import Pin
from motors import DCMotor, Servo
from scoop import Scoop

RPM_TO_RAD_S = 2 * math.pi / 60
DRIVE_RADIUS = 0.0837  # distance from centre to tracks in m
WHEEL_RADIUS = 0.0396  # radius of wheel in m

Led = Pin


def setup_actuators() -> tuple[DifferentialDrive, Scoop, Container, Led]:
    """Setup the motors"""
    left_drive_motor = DCMotor(
        motors.LEFT_MOTOR_PWM_PIN,
        motors.LEFT_MOTOR_DIR_PIN,
        motors.DRIVE_MAX_RPM * RPM_TO_RAD_S,
    )
    right_drive_motor = DCMotor(
        motors.RIGHT_MOTOR_PWM_PIN,
        motors.RIGHT_MOTOR_DIR_PIN,
        motors.DRIVE_MAX_RPM * RPM_TO_RAD_S,
    )
    scoop_motor = Servo(motors.SCOOP_MOTOR_PWM_PIN)
    vib_motor = DCMotor(
        motors.VIB_MOTOR_PWM_PIN,
        motors.VIB_MOTOR_DIR_PIN,
        motors.VIB_MAX_RPM * RPM_TO_RAD_S,
        freq=50_000,
    )
    container_servo = Servo(motors.CONTAINER_MOTOR_PWM_PIN)
    container = Container(container_servo)
    drive = DifferentialDrive(
        left_drive_motor, right_drive_motor, DRIVE_RADIUS, WHEEL_RADIUS
    )
    scoop = Scoop(scoop_motor, vib_motor)
    led = Pin(25, Pin.OUT)
    return drive, scoop, container, led


def handle_command(
    command: dict,
    drive: DifferentialDrive,
    scoop: Scoop,
    container: Container,
    led: Led,
) -> None:
    command_type = command.get("type")
    logger.info(f"Command: {command}")
    if command_type == "drive":
        v = command.get("v")
        w = command.get("w")
        if not isinstance(v, float) or not isinstance(v, float):
            logger.warning(f"Ignoring drive command with v: {v}, w: {w}")
            return

        drive.drive(v, w)
    elif command_type == "scoop":
        up = command.get("up")
        if not isinstance(up, bool):
            logger.warning(f"Ignoring scoop command with up: {up}")
            return

        scoop.up() if up else scoop.down()
    elif command_type == "container":
        open = command.get("open")
        if not isinstance(open, bool):
            logger.warning(f"Ignoring container command with open: {open}")
            return

        container.open() if open else container.close()
    elif command_type == "led":
        on = command.get("on")
        if not isinstance(on, bool):
            logger.warning(f"Ignoring LED command with on: {on}")
            return

        led.value(on)
    else:
        logger.warning(f"Received unknown command type: {command_type}")


def control(
    drive: DifferentialDrive,
    scoop: Scoop,
    container: Container,
    led: Led,
    commands: Queue,
) -> None:
    """Control thread for robot"""
    while True:
        success = False
        command = None
        while not success:
            success, command = commands.pop()  # type: ignore

        logger.info(f"Got command: {command}")
        if not isinstance(command, dict):
            logger.warning(f"Got command object: {type(command)} but expected dict")
            continue

        handle_command(command, drive, scoop, container, led)


def main():
    commands = Queue()
    drive, scoop, container, led = setup_actuators()
    comms = Comms(commands)
    comms.connect()
    _thread.start_new_thread(comms.run, ())
    try:
        control(drive, scoop, container, led, commands)
    except Exception as e:
        logger.warning(e)


if __name__ == "__main__":
    main()
