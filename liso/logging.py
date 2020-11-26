"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import logging

from .config import config


def create_logger():
    """General logger."""
    logger = logging.getLogger("LISO")

    fh = logging.FileHandler(config['DEFAULT']['LOG_FILE'],
                             mode='w', delay=True)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


logger = create_logger()
logger.setLevel(logging.INFO)


def create_opt_logger():
    """Logger for optimization information."""
    logger = logging.getLogger("LISO-Opt")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(config['DEFAULT']['OPT_LOG_FILE'],
                             mode="w", delay=True)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    return logger


opt_logger = create_opt_logger()
opt_logger.setLevel(logging.INFO)
