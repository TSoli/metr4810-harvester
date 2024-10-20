from queue import Queue

from comms import Comms
from logger import logger


def main() -> None:
    commands = Queue()
    comms = Comms(commands)
    comms.connect()
    logger.info(f"Connected")
    comms.run()


if __name__ == "__main__":
    main()
