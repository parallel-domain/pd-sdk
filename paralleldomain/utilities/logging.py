import logging
import sys
from typing import List

from coloredlogs import ColoredFormatter


def setup_loggers(logger_names: List[str], log_level: int = logging.INFO):
    for logger_name in logger_names:
        logger = logging.getLogger(name=logger_name)
        for handler in logger.handlers:
            logger.removeHandler(handler)
        logger.setLevel(log_level)
        formatter = ColoredFormatter(fmt="%(asctime)s %(name)s[%(funcName)s] %(levelname)s %(message)s")
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
