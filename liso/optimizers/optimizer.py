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

    def __init__(self, name):
        """Optimizer Class Initialization

        :param str name: Optimizer name.
        """
        self.name = name

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
        raise NotImplemented

    @abstractmethod
    def __str__(self):
        pass
