"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
from .elements import OperationalElement


class Covariable(OperationalElement):
    """Inherited from OperationalElement.

    Covariable is a variable which changes along with another variable.
    """
    def __init__(self, name, dependent, scale=1.0, shift=0.0):
        """Initialize CoVariable object

        The value of the variable is calculated by:
        covar = scale *var + shift

        :param dependent: string
            Name of the dependent variable.
        :param scale: float
            Coefficient.
        :param shift: float
            Coefficient.
        """
        super().__init__(name)

        self.dependent = dependent
        self.scale = scale
        self.shift = shift

    def list_item(self):
        return '{:12}  {:12}  {:^12.4e}  {:^12.4e}\n'.format(
            self.name[:12], self.dependent[:12], self.scale, self.shift)

    def __str__(self):
        text = '{:12}  {:12}  {:^12}  {:^12}\n'.format(
            'Name', 'Dependent', 'scale', 'shift') + self.list_item()
        return text
