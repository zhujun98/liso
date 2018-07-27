import logging

from .config import Config


def create_logger(name, filemode="w"):
    """"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(Config.LOG_FILENAME, mode=filemode)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def create_opt_logger():
    """Logger for optimization information."""
    logger = logging.getLogger("opt")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(Config.OPT_LOG_FILENAME, mode="w")
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    return logger


opt_logger = create_opt_logger()
