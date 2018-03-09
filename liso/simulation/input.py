"""
Author: Jun Zhu


The InputGenerator class is preserved to provide a high level interface
for producing input files for different simulations. An alternative
way is to use a template and replace some keywords in the template
each time.
"""
from abc import abstractmethod


class InputGenerator(object):
    def __init__(self):
        """Initialization."""

    @abstractmethod
    def add_quad(self):
        raise NotImplemented

    @abstractmethod
    def add_dipole(self):
        raise NotImplemented

    @abstractmethod
    def add_tws(self):
        raise NotImplemented

    @abstractmethod
    def add_gun(self):
        raise NotImplemented


class AstraInputGenerator(InputGenerator):
    pass


class ImpacttInputGenerator(InputGenerator):
    pass
