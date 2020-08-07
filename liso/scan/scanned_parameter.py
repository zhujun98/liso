"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
from ..elements import OperationalElement


class ScannedParameter(OperationalElement):
    def __init__(self, name, *, range=None, value=0.0, sigma=1.0, cutoff=2):
        """Initialization.

        :param tuple range:
            Range of the parameter.
        :param value: float
            Nominal value.
        :param sigma: float
            Standard deviation of the scan.
            Positive for absolute scan and negative for relative scan.
        :param cutoff: int
            Cutoff of the standard deviation.
        """
        super().__init__(name)

        self.value = value

        self._sigma = sigma

        if not isinstance(cutoff, int):
            raise TypeError("'cutoff' must be an integer!")
        self.cutoff = cutoff

    @property
    def sigma(self):
        if self._sigma < 0:
            return - self._sigma * self.value
        return self._sigma

    def list_item(self):
        return '{:12}  {:^12.4e}  {:^12.4e}  {:^12d}\n'.format(
            self.name[:12], self.value, self._sigma, self.cutoff)

    def __str__(self):
        return '{:12}  {:^12}  {:^12}  {:^12}\n'.format(
               'Name', 'Value', 'Std.', 'Cut-off') + \
               self.list_item()