#!/usr/bin/env python
"""
Optimizer class.

Author: Jun Zhu
"""
from abc import ABC, abstractmethod
import time

from ..logging import logger


class Optimizer(ABC):
    """Abstract class for optimizers.

    Attributes:
        name (str): Name of the optimizer.
        seed (int): Seed for random number. Default = None.
        printout (int): Level of printout.
    """
    category = None  # should be either 'global' or 'local'.
    name = None
    multiprocessing = False

    def __init__(self):
        """Optimizer Class Initialization."""
        self.seed = None
        if self.seed is None:
            self.seed = int(time.time())

    @abstractmethod
    def __call__(self, opt_problem):
        """Run Optimizer (Calling Routine)

        :param Optimization opt_problem: Optimization instance.

        :return: (optimized f,
                  optimized x,
                  miscellaneous information ready for printout).
        :rtype: (float, array-like, str)
        """
        pass

    @abstractmethod
    def __str__(self):
        pass

    def _check_workers(self, opt_prob):
        """Decide where to use multiprocessing.

        If the optimizer itself supports multiprocessing, the API will
        only use parallelized optimizer and serial version external
        codes. Otherwise, the API will use parallel version external
        codes.

        :param Optimization opt_problem: Optimization instance.
        """
        if self.multiprocessing is False:
            opt_prob.external_workers = opt_prob.workers
            opt_prob.workers = 1
        else:
            logger.info("Use parallelized {}.".format(self.name))
