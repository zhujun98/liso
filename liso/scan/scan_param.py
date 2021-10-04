"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import abc
from typing import Optional

import numpy as np

from ..elements import OperationalElement


class ScanParam(OperationalElement):
    """Base class for parameters used in parameter scan."""
    @abc.abstractmethod
    def generate(self, repeats: int = 1, cycles: int = 1):
        """Generate a sequence of parameters.

        :param repeats: Number of repeats of each element in the parameter
            space.
        :param cycles: Number of cycles of the sequence.

        For examples, generate(2, 3) = [1 1 2 2 1 1 2 2 1 1 2 2] with parameter
        space being [1 2].
        """
        raise NotImplementedError


class StepParam(ScanParam):
    """Generate parameters that change from start to end values stepwise."""
    def __init__(self, name: str, *, start: float, stop: float, num: int,
                 sigma: Optional[float] = 0.):
        """Initialization.

        :param start: The starting value of the scan.
        :param stop: The end value of the scan (included).
        :param num: Number of scanning points.
        :param sigma: Standard deviation of the jitter of the parameter if
            given. Positive for absolute jitter and negative for
            relative jitter.
        """
        super().__init__(name)

        self._start = start
        self._stop = stop

        if not isinstance(num, int) or num <= 0:
            raise ValueError("num must be a positive integer!")
        self._values = np.linspace(start, stop, num)

        self._sigma = sigma

    def __len__(self):
        return len(self._values)

    def _generate_once(self, repeats):
        """Override."""
        sigma = self._sigma
        ret = []
        for v in self._values:
            if sigma == 0.:
                ret.extend([v] * repeats)
            elif sigma < 0:
                ret.extend(v * (1 + np.random.normal(size=repeats) * sigma))
            else:
                ret.extend(v + np.random.normal(size=repeats) * sigma)
        return ret

    def generate(self, repeats=1, cycles=1):
        """Override."""
        ret = []
        for _ in range(cycles):
            ret.extend(self._generate_once(repeats))
        return np.array(ret)

    def list_item(self):
        """Override."""
        w = self._max_name_display_width
        return '{}  {:^12.4e}  {:^12.4e}  {:^12d}  {:^12.4e}\n'.format(
            self.name[:w].center(w), self._start, self._stop, len(self._values),
            self._sigma)

    def __str__(self):
        w = self._max_name_display_width
        return '{}  {:^12}  {:^12}  {:^12}  {:^12}\n'.format(
               'Name'.center(w), 'Start', 'Stop', 'Num', 'Sigma') + \
               self.list_item()


class SampleParam(ScanParam):
    """Generate parameters that are sampled uniformly within a given range."""
    def __init__(self, name: str, *, lb: float, ub: float):
        """Initialization.

        :param lb: The lower boundary of the sample.
        :param ub: The upper boundary of the sample (not included).
        """
        super().__init__(name)

        if lb > ub:
            lb, ub = ub, lb
        self._lb = lb
        self._ub = ub

    def __len__(self):
        return 1

    def generate(self, repeats=1, cycles=1):
        """Override."""
        return np.random.uniform(self._lb, self._ub, repeats * cycles)

    def list_item(self):
        """Override."""
        w = self._max_name_display_width
        return '{}  {:^12.4e}  {:^12.4e}\n'.format(
            self.name[:w].center(w), self._lb, self._ub)

    def __str__(self):
        w = self._max_name_display_width
        return '{}  {:^12}  {:^12}\n'.format(
            'Name'.center(w), 'Lower bound', 'Upper bound') + self.list_item()


class JitterParam(ScanParam):
    """Generate parameters that are randomly sampled around a given value."""
    def __init__(self, name: str, *, value: float, sigma: Optional[float] = 0.):
        """Initialization.

        :param value: The reference value.
        :param sigma: Standard deviation of the jitter of the parameter if
            given. Positive for absolute jitter and negative for
            relative jitter.
        """
        super().__init__(name)

        self._value = value
        self._sigma = sigma

    def __len__(self):
        return 1

    def generate(self, repeats=1, cycles=1):
        """Override."""
        rand_nums = np.random.normal(size=repeats * cycles)
        if self._sigma < 0:
            return self._value * (1 + rand_nums * self._sigma)
        return self._value + rand_nums * self._sigma

    def list_item(self):
        """Override."""
        w = self._max_name_display_width
        return '{}  {:^12.4e}  {:^12.4e}\n'.format(
            self.name[:w].center(w), self._value, self._sigma)

    def __str__(self):
        w = self._max_name_display_width
        return '{}  {:^12}  {:^12}\n'.format(
            'Name'.center(w), 'Value', 'Sigma') + self.list_item()
