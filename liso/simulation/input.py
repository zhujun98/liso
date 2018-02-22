"""
Author: Jun Zhu

"""
from abc import abstractmethod


class Input(object):
    def __init__(self, input_file):
        """Initialization."""
        self.input_file = None
        self.template = None
        self.charge = None

    @abstractmethod
    def generate(self):
        raise NotImplemented


class AstraInput(Input):
    def generate(self):
        raise NotImplemented
