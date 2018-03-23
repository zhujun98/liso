#!/usr/bin/env python
"""
Objective class.

Author: Jun Zhu
"""
from .passive_optimization_element import PassiveOptimizationElements
from ..config import Config

INF = Config.INF


class Objective(PassiveOptimizationElements):
    """Optimization Objective Class"""
    def __init__(self, name, expr=None, scale=1.0, func=None, optimum=-INF):
        """Initialization."""
        super().__init__(name, expr=expr, scale=scale, func=func)
        self.value = INF
        self.optimum = optimum

    def __repr__(self):
        return '{:^12}  {:^12.4e}  {:^12.4e}\n'.format(
            self.name[:12], self.value, self.optimum)

    def __str__(self):
        return '{:^12}  {:^12}  {:^12}\n'.format(
            'Name', 'Value', 'Optimum') + self.__repr__()
