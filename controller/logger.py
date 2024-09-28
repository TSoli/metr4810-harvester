import logging
import os
import sys
from datetime import datetime as dt

if not os.path.exists("logs"):
    os.makedirs("logs")

now = dt.now().strftime("%Y%m%d_%H%M_%S")
logfile = os.path.join("logs", f"{now}_controller.log")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(logfile), logging.StreamHandler(sys.stdout)],
)
# TODO: delete old logs

logger = logging.getLogger("ControllerLogger")
