import logging
from datetime import datetime as dt

now = dt.now().strftime("%Y%m%d_%H%M_%S")
logging.basicConfig(
    filename=f"{now}_robot.log",
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
# TODO: delete old logs

logger = logging.getLogger("RobotLogger")
