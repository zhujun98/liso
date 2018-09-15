"""
Author: Jun Zhu, zhujun981661@gmail.com
"""
from abc import ABC, abstractmethod


class Operation(ABC):
    """Abstract class for Optimization and Jitter.

    Attributes:
        name (str): Name of the operation.
        printout (int): Level of printout.
    """
    def __init__(self, name):
        """Initialization.

        :param (str) name: Name of the operation (arbitrary).
        """
        self.name = name
        self._workers = None
        self.printout = 0

    @property
    def workers(self):
        return self._workers

    @workers.setter
    def workers(self, value):
        raise SystemError(
            "For parrallel run, you should set workers for Beamlines, for example:\n"
            "linac.add_beamline('astra',\n"
            "                   name='gun',\n"
            "                   fin='astra_injector/injector.in',\n"
            "                   template='astra_advanced/injector.in.000',\n"
            "                   pout='injector.0450.001',\n"
            "                   workers=12)\n"
        )

    @abstractmethod
    def __str__(self):
        pass
