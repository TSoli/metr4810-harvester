import logging

import uio
import uos


class StreamToLogger(uio.IOBase):
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def write(self, data):
        data = data.decode()
        if data == "\r\n":
            return 0

        self.logger.info(data)
        return len(data)

    def read(self, n=-1):
        return ""  # No-op for read, as we're only handling writes

    def readinto(self, buf):
        # No-op readinto; return 0 to indicate no bytes read, making it compatible with uos.dupterm
        return 0


def _setup_logger() -> logging.Logger:
    if not "logs" in uos.listdir():
        uos.mkdir("logs")

    logs = uos.listdir("logs")
    if len(logs) == 0:
        log_num = 0
    else:
        log_nums = [int(log[:2]) for log in logs]
        log_num = sorted(log_nums)[-1] + 1

    filename = f"logs/{log_num:02d}_robot.log"
    logging.basicConfig(
        filename=filename,
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("RobotLogger")

    uos.dupterm(StreamToLogger(logger))
    return logger


logger = _setup_logger()
