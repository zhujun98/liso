"""
Author: Jun Zhu, zhujun981661@gmail.com
"""
from abc import ABC, abstractmethod


class Operation(ABC):
    """Abstract class for Optimization and Jitter.

    Attributes:
        name (str): Name of the operation.
        workers (int): Number of processes for parallel accelerator codes.
        printout (int): Level of printout.
    """
    def __init__(self, name):
        """Initialization.

        :param (str) name: Name of the operation (arbitrary).
        """
        self.name = name
        self.workers = 1
        self.printout = 0

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
