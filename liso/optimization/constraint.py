"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""

# In the optimization problem definition, the equality and inequality
# constraints are defined as:
#
#     g_j(x) = 0, j = 1, ..., m_e
#
#     g_j(x) <= 0, j = m_e + 1, ..., m
#
# respectively. We must convert the constraint defined by the client to
# the standard constraint.

import math

from ..elements import EvaluatedElement
from ..logging import logger, opt_logger


class IConstraint(EvaluatedElement):
    """Inequality constraint class. Inherited from EvaluatedElement."""
    def __init__(self, name, expr=None, scale=1.0, func=None, **kwargs):
        """Initialization."""
        super().__init__(name, expr=expr, scale=scale, func=func)

        self._value = None  # the real value
        self.value = math.inf  # the value seen by the optimizer
        # lb and ub are both properties, only one of them is allowed to be
        # specified since all inequality constraint must be normalized to
        # [-INF, 0]. Namely, if lb is specified, then ub will be reset to
        # INF; however, if ub is specified, then lb will be reset to -INF.
        self._lb = -math.inf
        self.lb = -math.inf
        self._ub = 0.0
        self.ub = 0.0

        for key in kwargs:
            if key.lower() == 'lb':
                self.lb = kwargs[key]
            elif key.lower() == 'ub':
                self.ub = kwargs[key]
            else:
                raise ValueError("Unknown keyword argument!")

        if len(kwargs) > 1:
            info = "Constraint '{}': 'lb' is ignored since 'ub' is specified!"\
                   .format(self.name)
            logger.warning(info)
            opt_logger.warning(info)

    @property
    def ub(self):
        return self._ub

    @ub.setter
    def ub(self, value):
        self._ub = value
        self._lb = -math.inf
        self._value = math.inf  # reset _value to upset the inequality condition

    @property
    def lb(self):
        return self._lb

    @lb.setter
    def lb(self, value):
        self._lb = value
        self._ub = math.inf
        self._value = -math.inf  # reset _value to upset the inequality condition

    @property
    def value(self):
        # The value is normalized to(-INF, 0]
        if self.ub == math.inf:
            return self._lb - self._value
        if self.lb == -math.inf:
            return self._value - self._ub
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


class EConstraint(EvaluatedElement):
    """Equality constraint class. Inherited from EvaluatedElement."""
    def __init__(self, name, expr=None, scale=1.0, func=None, eq=0.0):
        """Initialization."""
        super().__init__(name, expr=expr, scale=scale, func=func)
        self.eq = eq
        self._value = None  # the real value
        self.value = math.inf  # the value seen by the optimizer

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
