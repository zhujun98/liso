"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
from abc import ABC, abstractmethod
import time


class Optimizer(ABC):
    """Abstract class for optimizers.

    Attributes:
        name (str): Name of the optimizer.
        seed (int): Seed for random number. Default = None.
    """
    category = None  # should be either 'global' or 'local'.
    name = None
    multiprocessing = False

    def __init__(self, seed=None):
        """Optimizer Class Initialization."""
        self._seed = int(time.time()) if seed is None else int(seed)

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
