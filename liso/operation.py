"""
Author: Jun Zhu, zhujun981661@gmail.com
"""
from abc import ABC, abstractmethod


class Operation(ABC):
    """Abstract class for Optimization and Jitter.

    Attributes:
        name (str): Name of the operation.
        printout (int): Level of printout.
        monitor_time (bool):
    """
    def __init__(self, name):
        """Initialization.

        :param (str) name: Name of the operation (arbitrary).
        """
        self.name = name
        self._workers = 1  # number of threads
        self.printout = 0
        self.monitor_time = False

    @property
    def workers(self):
        return self._workers

    @workers.setter
    def workers(self, value):
        if isinstance(value, int) and value > 0:
            self._workers = value

    @abstractmethod
    def __str__(self):
        pass
