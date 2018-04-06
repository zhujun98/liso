#!/usr/bin/env python
"""
Jitter class

Author: Jun Zhu

TODO: For now the cutoff takes no effect!
"""
from ..elements import OperationalElement


class Jitter(OperationalElement):
    """Inherited from OperationalElement."""
    def __init__(self, name, *, value=0.0, sigma=0.0, cutoff=2):
        """Initialization.

        :param value: float
            Nominal value.
        :param sigma: float
            Standard deviation of the jitter.
            Positive for absolute jitter and negative for relative jitter.
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
