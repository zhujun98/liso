#!/usr/bin/env python
"""
Constraint class

In the optimization problem definition, the equality and inequality
constraints are defined as:

    g_j(x) = 0, j = 1, ..., m_e

    g_j(x) <= 0, j = m_e + 1, ..., m

respectively. We must convert the constraint defined by the client to
the standard constraint.

Author: Jun Zhu
"""
import warnings

from .descriptive_parameter import DescriptiveParameter
from ..config import Config

INF = Config.INF


class IConstraint(DescriptiveParameter):
    """Optimization inequality constraint class."""
    def __init__(self, name, expr=None, scale=1.0, func=None, **kwargs):
        """Initialization."""
        super().__init__(name, expr=expr, scale=scale, func=func)

        self._lb = -INF
        self._ub = 0.0

        self._value = None  # the real value
        self.value = None  # the value seen by the optimizer

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
        self._value = INF

    @property
    def lb(self):
        return self._lb

    @lb.setter
    def lb(self, value):
        self._lb = value
        self._ub = INF
        self.value = -INF

    @property
    def value(self):
        # The value is normalized to(-INF, 0]
        if self.ub == INF:
            return self.lb - self._value
        if self.lb == -INF:
            return self._value - self.ub
        raise ValueError("Wrong boundary values in IConstraint!")

    @value.setter
    def value(self, v):
        self._value = v

    def list_item(self):
        return '{:12}  {:^12.4e}  {:^12.4e}  {:^12.4e}\n'.format(
               self.name[:12], self._value, self.lb, self.ub)

    def __str__(self):
        return '{:12}  {:^12}  {:^12}  {:^12}\n'.format(
               'Name', 'Value', 'Lower Bound', 'Upper Bound') + \
               self.list_item()


class EConstraint(DescriptiveParameter):
    """Optimization equality constraint class."""
    def __init__(self, name, expr=None, scale=1.0, func=None, eq=0.0):
        """Initialization."""
        super().__init__(name, expr=expr, scale=scale, func=func)
        self.eq = eq
        self._value = None  # the real value
        self.value = INF  # the value seen by the optimizer

    @property
    def value(self):
        # The value is normalized to ~0.0
        return self._value - self.eq

    @value.setter
    def value(self, v):
        self._value = v

    def list_item(self):
        return '{:12}  {:^12.4e}  {:^12.4e}\n'.format(
               self.name[:12], self._value, self.eq)

    def __str__(self):
        return '{:12}  {:^12}  {:^12}\n'.format('Name', 'Value', 'Equal') + \
               self.list_item()
