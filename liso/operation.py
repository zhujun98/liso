"""
Author: Jun Zhu
"""
from abc import ABC, abstractmethod


class Operation(ABC):
    """Abstract class for Optimization and Jitter."""

    def __init__(self, name):
        """Initialization.

        :param name: str
            Name of the optimization problem (arbitrary).
        """
        self.name = name
        self._workers = 1  # number of threads
        self.printout = 0  # Level of printout
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
