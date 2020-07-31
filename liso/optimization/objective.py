"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import math

from ..elements import EvaluatedElement


class Objective(EvaluatedElement):
    """Objective Class. Inherited from EvaluatedElement"""
    def __init__(self, name, expr=None, scale=1.0, func=None, optimum=-math.inf):
        """Initialization."""
        super().__init__(name, expr=expr, scale=scale, func=func)
        self.value = math.inf
        self.optimum = optimum

    def list_item(self):
        return '{:12}  {:^12.4e}  {:^12.4e}\n'.format(
            self.name[:12], self.value, self.optimum)

    def __str__(self):
        return '{:12}  {:^12}  {:^12}\n'.format(
            'Name', 'Value', 'Optimum') + self.list_item()
