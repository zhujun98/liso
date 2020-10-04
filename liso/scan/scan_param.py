"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import itertools

import numpy as np

from ..elements import OperationalElement


class ScanParam(OperationalElement):
    def __init__(self, name, start, stop=None, num=1, *, sigma=0.):
        """Initialization.

        :param float start: the starting value of the scan.
        :param float/None stop: the end value of the scan (included).
        :param int num: number of scanning points.
        :param float sigma: standard deviation of the jitter of the
            parameter if given. Positive for absolute jitter and negative
            for negative jitter.
        """
        super().__init__(name)

        if not isinstance(num, int) or num <= 0:
            raise ValueError("num must be a positive integer!")
        self._num = num

        self._start = start
        if stop is None:
            self._stop = start
            self._num = 1
        else:
            self._stop = stop

        self._sigma = sigma

        self._count = 0
        self._values = np.linspace(self._start, self._stop, self._num)

    def __len__(self):
        return self._num

    def __iter__(self):
        return self

    def __next__(self):
        i = self._count
        sigma = self._sigma
        if i < self._num:
            self._count += 1
            if sigma == 0.:
                return self._values[i]
            if sigma < 0:
                return self._values[i] * (1 + np.random.normal() * sigma)
            return self._values[i] + np.random.normal() * sigma

        raise StopIteration

    def repeat(self, times=1):
        ret = []
        self._count = 0
        while self._count < self._num:
            for _ in range(times):
                ret.append(self.__next__())
                self._count -= 1
            self._count += 1
        self._count = 0  # reset iterator
        return ret

    def cycle(self, times=1, repeat=1):
        ret = []
        for _ in range(times):
            ret.extend(self.repeat(repeat))
        return ret

    def list_item(self):
        """Override."""
        return '{:12}  {:^12.4e}  {:^12.4e}  {:^12d}  {:^12.4e}\n'.format(
            self.name[:12], self._start, self._stop, self._num, self._sigma)

    def __str__(self):
        return '{:12}  {:^12}  {:^12}  {:^12}  {:^12}\n'.format(
               'Name', 'Start', 'Stop', 'Num', 'Sigma') + \
               self.list_item()
