#!/usr/bin/env python
"""
Objective class.

Author: Jun Zhu
"""
from .descriptive_parameter import DescriptiveParameter
from ..config import Config

INF = Config.INF


class Objective(DescriptiveParameter):
    """Optimization Objective Class"""
    def __init__(self, name, expr=None, scale=1.0, func=None, optimum=-INF):
        """Initialization."""
        super().__init__(name, expr=expr, scale=scale, func=func)
        self.value = INF
        self.optimum = optimum

    def list_item(self):
        return '{:12}  {:^12.4e}  {:^12.4e}\n'.format(
            self.name[:12], self.value, self.optimum)

    def __str__(self):
        return '{:12}  {:^12}  {:^12}\n'.format(
            'Name', 'Value', 'Optimum') + self.list_item()
