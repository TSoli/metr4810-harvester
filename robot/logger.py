import logging

import uos


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

    return logging.getLogger("RobotLogger")


logger = _setup_logger()
