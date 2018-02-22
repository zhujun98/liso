"""
Author: Jun Zhu

"""
import numpy as np


class Stats(object):
    """Store the statistic values of an array-like object.

    Attributes
    ----------
    start: float
        First value.
    end: float
        Last value.
    max: float
        Maximum value.
    min: float
        Minimum value.
    ave: float
        Average value.
    std: float
        Standard deviation.
    """
    def __init__(self):
        """"""
        self.start = None
        self.end = None
        self.max = None
        self.min = None
        self.ave = None
        self.std = None

    def __str__(self):
        """"""
        text = "{:12}    {:12}    {:12}    {:12}    {:12}    {:12}\n".\
            format('start', 'end', 'minimum', 'maximum', 'average', 'std')

        text += "{:12.4e}    {:12.4e}    {:12.4e}    {:12.4e}    {:12.4e}    {:12.4e}\n\n".\
            format(self.start, self.end, self.min, self.max, self.ave, self.std)

        return text

    def update(self, data):
        """Update attributes

        :param data: array-like
            Input data.
        """
        data = np.asarray(data)
        if data.ndim > 1:
            raise ValueError("One-dimensional array is foreseen!")

        self.start = data[0]
        self.end = data[-1]
        self.max = data.max()
        self.min = data.min()
        self.ave = data.mean()
        self.std = data.std(ddof=0)
