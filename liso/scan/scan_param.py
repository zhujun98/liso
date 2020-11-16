"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import abc

import numpy as np

from ..elements import OperationalElement


class _IterParam(OperationalElement):
    """Base class for parameters used in parameter scan."""
    def __init__(self, name):
        super().__init__(name)

    @abc.abstractmethod
    def generate(self, repeats=1, cycles=1):
        """Generate a sequence of parameters.

        :param int repeats: number of repeats of each element in the parameter
            space.
        :param int cycles: number of cycles of the sequence.

        For examples, generate(2, 3) = [1 1 2 2 1 1 2 2 1 1 2 2] with parameter
        space being [1 2].
        """
        raise NotImplementedError


class ScanParam(OperationalElement):
    """ScanParam class.

    A scan parameter is a parameter that changes from start to end
    values stepwise.
    """
    def __init__(self, name, *, start, stop, num, sigma=0.):
        """Initialization.

        :param float start: the starting value of the scan.
        :param float stop: the end value of the scan (included).
        :param int num: number of scanning points.
        :param float sigma: standard deviation of the jitter of the
            parameter if given. Positive for absolute jitter and negative
            for relative jitter.
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
        return '{:12}  {:^12.4e}  {:^12.4e}  {:^12d}  {:^12.4e}\n'.format(
            self.name[:12], self._start, self._stop, len(self._values),
            self._sigma)

    def __str__(self):
        return '{:12}  {:^12}  {:^12}  {:^12}  {:^12}\n'.format(
               'Name', 'Start', 'Stop', 'Num', 'Sigma') + \
               self.list_item()


class SampleParam(OperationalElement):
    """SampleParam class.

    A sample parameter is a parameter that is sampled uniformly within a
    given range.
    """
    def __init__(self, name, *, lb, ub):
        """Initialization.

        :param float lb: the lower boundary of the sample.
        :param float ub: the upper boundary of the sample (not included).
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
        return '{:12}  {:^12.4e}  {:^12.4e}\n'.format(
            self.name[:12], self._lb, self._ub)

    def __str__(self):
        return '{:12}  {:^12}  {:^12}\n'.format(
            'Name', 'Lower bound', 'Upper bound') + self.list_item()


class JitterParam(OperationalElement):
    """JitterParam class.

    A jitter parameter is a parameter that jitters around a given value.
    """
    def __init__(self, name, *, value, sigma=0.):
        """Initialization.

        :param float value: the reference value.
        :param float sigma: standard deviation of the jitter of the
            parameter if given. Positive for absolute jitter and negative
            for negative jitter.
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
        return '{:12}  {:^12.4e}  {:^12.4e}\n'.format(
            self.name[:12], self._value, self._sigma)

    def __str__(self):
        return '{:12}  {:^12}  {:^12}\n'.format('Name', 'Value', 'Sigma') + \
               self.list_item()
