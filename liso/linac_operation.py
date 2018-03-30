"""
Author: Jun Zhu
"""
from abc import ABC, abstractmethod

from .simulation import Linac


class LinacOperation(ABC):
    """Abstract class for LinacOptimization and LinacJitter."""

    def __init__(self, linac, *, name):
        """Initialization.

        :param linac: Linac object
            Linac instance.
        :param name: str
            Name of the optimization problem (arbitrary).
        """
        if isinstance(linac, Linac):
            self._linac = linac
        else:
            raise TypeError("{} is not a Linac instance!".format(linac))

        self.name = name
        self._workers = 1
        self.verbose = False
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
