#!/usr/bin/env python
"""
Constraint class

Author: Jun Zhu
"""
import warnings

from .passive_optimization_element import PassiveOptimizationElements
from ..config import Config

INF = Config.INF


class IConstraint(PassiveOptimizationElements):
    """Optimization inequality constraint class."""
    def __init__(self, name, expr=None, scale=1.0, func=None, **kwargs):
        """Initialization."""
        super().__init__(name, expr=expr, scale=scale, func=func)

        self._lb = -INF
        self._ub = 0.0

        for key in kwargs:
            if key.lower() == 'lb':
                self.lb = kwargs[key]
            elif key.lower() == 'ub':
                self.ub = kwargs[key]
            else:
                raise ValueError("Unknown keyword argument!")

        if len(kwargs) > 1:
            warnings.warn("'lb' is ignored since 'ub' is specified!")

    @property
    def ub(self):
        return self._ub

    @ub.setter
    def ub(self, value):
        self._ub = value
        self._lb = -INF
        self.value = INF

    @property
    def lb(self):
        return self._lb

    @lb.setter
    def lb(self, value):
        self._lb = value
        self._ub = INF
        self.value = -INF

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
