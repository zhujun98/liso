"""
Author: Jun Zhu, zhujun981661@gmail.com
"""
from abc import ABC, abstractmethod


class Operation(ABC):
    """Abstract class for Optimization and Jitter.

    Attributes:
        name (str): Name of the operation.
        workers (int): Number of processes for parallel run.
        external_workers (int): Number of processes for external codes, for
            example, ASTRA.
        printout (int): Level of printout.
    """
    def __init__(self, name):
        """Initialization.

        :param (str) name: Name of the operation (arbitrary).
        """
        self.name = name
        self._workers = None
        self.workers = 1
        self.external_workers = 1
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
