"""
Author: Jun Zhu
"""
from abc import ABC, abstractmethod


class OperationElement(ABC):
    """Abstract class."""
    def __init__(self, name):
        """Initialization.

        :param name: str
            Name of the element.
        """
        self.name = name

    @abstractmethod
    def list_item(self):
        pass

    @abstractmethod
    def __str__(self):
        pass
