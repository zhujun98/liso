"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""


class Normalizer:
    """Normalize and unnormalize data and track the record."""

    def __init__(self, data, *, minmax=None):
        """Initialization.

        :param pandas.DataFrame/dict data: reference data to be normalized.
            It should support __getitem__ or __setitem__.
        :param dict minmax: a dictionary of (lower boundary, higher boundary)
            of the MinMax normalizer.
        """
        self._minmax = dict()
        if minmax is not None:
            for key, item in minmax.items():
                if len(item) != 2:
                    raise ValueError(
                        f"MinMax normalize must only have two parameters: "
                        f"{len(item)}!")
                self._minmax[key] = tuple(item)

        for key in data:
            if key not in self._minmax:
                self._minmax[key] = data[key].min(), data[key].max()
            self._normalize_minmax(data, key)

    def normalize(self, data):
        """Normalize the data inplace.

        :param pandas.DataFrame/dict data: data to be normalized.
            It should support __getitem__ or __setitem__.
        """
        for key in data:
            if key not in self._minmax:
                raise KeyError(f"{key} is not found in the data!")

            if key in self._minmax:
                self._normalize_minmax(data, key)

    def unnormalize(self, data):
        """Unnormalize the data inplace.

        :param pandas.DataFrame/dict data: data to be normalized.
            It should support __getitem__ or __setitem__.
        """
        for key in data:
            if key not in self._minmax:
                raise KeyError(f"{key} is not found in the data!")

            if key in self._minmax:
                self._unnormalize_minmax(data, key)

    def _normalize_minmax(self, data, key):
        lb, ub = self._minmax[key]
        data[key] = 2 * (data[key] - lb) / (ub - lb) - 1

    def _unnormalize_minmax(self, data, key):
        lb, ub = self._minmax[key]
        data[key] = 0.5 * (data[key] + 1) * (ub - lb) + lb
