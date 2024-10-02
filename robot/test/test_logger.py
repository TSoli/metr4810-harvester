import usys
from logger import logger

logger.critical("About to die")
print("Hello")
print("Error?", file=usys.stderr)
raise NotImplementedError
