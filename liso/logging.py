import logging

from .config import config


def create_logger():
    """General logger."""
    logger = logging.getLogger("LISO")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(config['DEFAULT']['LOG_FILE'], mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


logger = create_logger()


def create_opt_logger():
    """Logger for optimization information."""
    logger = logging.getLogger("LISO-Opt")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(config['DEFAULT']['OPT_LOG_FILE'], mode="w")
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    return logger


opt_logger = create_opt_logger()
