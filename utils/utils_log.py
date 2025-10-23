import logging
from typing import Union

logger = logging.getLogger("pysr")
logger.handlers[0].setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s"))

def log_info(msg: Union[str, int]):
    """Function to log information, call to this function ensures that the BasicConfig & logging level are respected"""
    logger.info(msg)
