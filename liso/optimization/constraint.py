#!/usr/bin/env python
"""
Constraint class

Author: Jun Zhu
"""
from .passive_optimization_element import PassiveOptimizationElements

from ..config import Config

INF = Config.INF


class IConstraint(PassiveOptimizationElements):
    """Optimization inequality constraint class."""
    def __init__(self, name, expr=None, scale=1.0, func=None, lb=-INF, ub=0.0):
        """Initialization."""
        super().__init__(name, expr=expr, scale=scale, func=func)
        self.lb = lb
        self.ub = ub

    def __repr__(self):
        return '{:^12}  {:^12.4e}  {:^12.4e}  {:^12.4e}\n'.format(
               self.name[:12], self.value, self.lb, self.ub)

    def __str__(self):
        return '{:^12}  {:^12}  {:^12}  {:^12}\n'.format(
               'Name', 'Value', 'Lower Bound', 'Upper Bound') + \
               self.__repr__()


class EConstraint(PassiveOptimizationElements):
    """Optimization equality constraint class."""
    def __init__(self, name, expr=None, scale=1.0, func=None, eq=0.0):
        """Initialization."""
        super().__init__(name, expr=expr, scale=scale, func=func)
        self.eq = eq

    def __repr__(self):
        return '{:^12}  {:^12.4e}  {:^12.4e}\n'.format(
               self.name[:12], self.value, self.eq)

    def __str__(self):
        return '{:^12}  {:^12}  {:^12}\n'.format('Name', 'Value', 'Equal') + \
               self.__repr__()
