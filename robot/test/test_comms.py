from comms import Comms
from logger import logger


def main() -> None:
    comms = Comms()
    comms.connect()
    logger.info(f"Connected")
    comms.run()


if __name__ == "__main__":
    main()
