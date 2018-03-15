#!/usr/bin/env python
"""
Constraint class

Author: Jun Zhu
"""
from abc import ABC
from abc import abstractmethod

from ..config import Config

INF = Config.INF


class Constraint(ABC):
    """Optimization Constraint Class"""
    def __init__(self, name):
        """Constraint class initialization

        :param name: string
            Variable Name.
        """
        self.name = name
        self.value = 0.0

    @abstractmethod
    def __repr__(self):
        raise NotImplemented

    @abstractmethod
    def __str__(self):
        raise NotImplemented


class IConstraint(Constraint):
    """Equality constraint class."""
    def __init__(self, name, lb=-INF, ub=0.0):
        """Initialization."""
        super().__init__(name)
        self.lb = lb
        self.ub = ub

    def __repr__(self):
        return '{:^12}  {:^12.4e}  {:^12.4e}  {:^12.4e}\n'.format(
               self.name[:12], self.value, self.lb, self.ub)

    def __str__(self):
        return '{:^12}  {:^12}  {:^12}  {:^12}\n'.format(
               'Name', 'Value', 'Lower Bound', 'Upper Bound') + \
               self.__repr__()


class EConstraint(Constraint):
    """Equality constraint class."""
    def __init__(self, name, eq=0.0):
        """Initialization."""
        super().__init__(name)
        self.eq = eq

    def __repr__(self):
        return '{:^12}  {:^12.4e}  {:^12.4e}\n'.format(
               self.name[:12], self.value, self.eq)

    def __str__(self):
        return '{:^12}  {:^12}  {:^12}\n'.format('Name', 'Value', 'Equal') + \
               self.__repr__()
