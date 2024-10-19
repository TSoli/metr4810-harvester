from queue import Queue

from comms import Comms
from logger import logger


def main() -> None:
    commands = Queue()
    comms = Comms(commands)
    comms.connect()
    logger.info(f"Connected")
    try:
        comms.run()
    except Exception as e:
        logger.info(e)


if __name__ == "__main__":
    main()
