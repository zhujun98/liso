#!/usr/bin/env python
"""
Optimizer class.

Author: Jun Zhu
"""
from abc import ABC, abstractmethod
import time


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
        self._workers = None
        self.workers = 1

    @property
    def workers(self):
        return self._workers

    @workers.setter
    def workers(self, value):
        if isinstance(value, int) and value > 0:
            if value > 1 and not self.multiprocessing:
                raise ValueError("{} does not support parallel optimization!".
                                 format(self.__class__.__name__))
            self._workers = value
        else:
            raise ValueError("Invalid input {} for 'workers'!".format(value))

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
