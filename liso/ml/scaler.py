"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import numpy as np


class Normalizer:
    """Normalize and unnormalize data and track the record."""

    _transformers = {
        'log10': lambda x, y=0.: np.log10(x + y),
        'log': lambda x, y=0.: np.log(x + y),
        'sqrt': lambda x, y=0.: np.sqrt(x + y),
        'cbrt': lambda x, y=0.: np.cbrt(x + y),
    }

    _inv_transformers = {
        'log10': lambda x, y=0.: np.power(10., x) - y,
        'log': lambda x, y=0.: np.exp(x) - y,
        'sqrt': lambda x, y=0.: np.square(x) - y,
        'cbrt': lambda x, y=0.: np.power(x, 3) - y,
    }

    def __init__(self, data, *, minmax=None, transform=None):
        """Initialization.

        :param pandas.DataFrame/dict data: reference data to be normalized.
            It should support __getitem__ or __setitem__. The items within
            the data set will be normalized individually.
        :param dict minmax: a dictionary of (lower boundary, upper boundary)
            of the MinMax normalizer. Note that the lower and upper boundaries
            do not have to be the min and max values of the corresponding
            item in the data if given.
        :param dict transform: a dictionary of transformation to be applied
            before MinMax normalization. The value should be in the format of
            method or (method, shift) or (method, shift, flip), where method
            can be 'log10', 'log', 'sqrt', and 'cbrt'. The default
            value of shift and flip are 0 and False, respectively, if not given.
            flip should be used to convert left-skewed data to a right-skewed
            one.
        """
        self._tran = dict()
        if transform is not None:
            for key, item in transform.items():
                flip = False
                shift = 0.
                if not isinstance(item, str):
                    if len(item) == 2:
                        item, shift = item
                    else:
                        item, shift, flip = item
                        flip = bool(flip)

                if item not in self._transformers:
                    raise ValueError(f"Unknown transformer: {item}")

                self._tran[key] = (item, shift, flip)

        self._minmax = dict()
        if minmax is not None:
            for key, item in minmax.items():
                if len(item) != 2:
                    raise ValueError(
                        f"MinMax normalize must only have two parameters: "
                        f"{len(item)}!")
                lb, ub = item
                if lb >= ub:
                    raise ValueError(f"Upper boundary must be larger than"
                                     f"lower boundary: {key: ({lb}, {ub})}")

                self._minmax[key] = (lb, ub)

        for key in data:
            if key not in self._minmax:
                # 'flip' affects automatically generated min/max boundary
                flip = False
                if key in self._tran:
                    _, _, flip = self._tran[key]

                if flip:
                    self._minmax[key] = -data[key].max(), -data[key].min()
                else:
                    self._minmax[key] = data[key].min(), data[key].max()

            if key in self._tran:
                self._transform(data, key)

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

            if key in self._tran:
                self._inv_transform(data, key)

    def _normalize_minmax(self, data, key):
        lb, ub = self._minmax[key]
        data[key] = 2.0 * (data[key] - lb) / (ub - lb) - 1.

    def _unnormalize_minmax(self, data, key):
        lb, ub = self._minmax[key]
        data[key] = 0.5 * (data[key] + 1.) * (ub - lb) + lb

    def _transform(self, data, key):
        method, shift, flip = self._tran[key]
        f = self._transformers[method]
        lb, ub = self._minmax[key]
        self._minmax[key] = f(lb, shift), f(ub, shift)
        data[key] = f(-data[key] if flip else data[key], shift)

    def _inv_transform(self, data, key):
        method, shift, flip = self._tran[key]
        data[key] = self._inv_transformers[method](data[key], shift)
        if flip:
            data[key] = - data[key]
