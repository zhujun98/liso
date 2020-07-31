"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import math

import numpy as np

from ..elements import EvaluatedElement


class Response(EvaluatedElement):
    """Inherited from OperationalElement.

    A Response instance records the responses of a parameter to
    the jitters.
    """
    def __init__(self, name, *, expr=None, scale=1.0):
        """Initialization."""
        # 'func' is for now not allowed since it is not necessary.
        super().__init__(name, expr=expr, scale=scale, func=None)
        self.values = list()

    @property
    def sigma(self):
        if not self.values:
            return math.inf
        return np.std(self.values)

    def list_item(self):
        return '{:12}  {:18}  {:^12.4e}\n'.format(
            self.name[:12], '.'.join(self.expr)[:18], self.sigma)

    def __str__(self):
        return '{:12}  {:18}  {:^12}\n'.format('Name', 'Expression', 'Std.')\
               + self.list_item()
