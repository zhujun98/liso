#!/usr/bin/python
"""
Helper functions for visualization package.

Author: Jun Zhu
"""
import random
import re

import numpy as np
from scipy.ndimage.filters import gaussian_filter


def sample_data(x, y, *, bins=None, sigma=None, sample=20000):
    """Sample the data and calculate the density map.

    :param x: pandas.Series
        x data.
    :param y: pandas.Series
        y data.
    :param bins: int or (int, int)
        No. of bins used in numpy.histogram2d().
    :param sigma: numeric
        Standard deviation of Gaussian kernel of the Gaussian filter.
    :param sample: scalar >=0
        If sample < 1.0, sample by fraction;
        else, sample by count (round to integer).

    :returns x_sample: pandas.Series
        sampled x data.
    :returns y_sample: pandas.Series
        sampled y data
    :returns z: numpy.ndarray.
        Normalized density at each sample point.
    """
    if int(sample) > 0:
        n = int(sample)
    elif sample > 0:
        n = int(sample * len(x))
    else:
        raise ValueError("Negative sample value!")

    H, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    x_center = (x_edges[1:] + x_edges[0:-1]) / 2
    y_center = (y_edges[1:] + y_edges[0:-1]) / 2
    H_blurred = gaussian_filter(H, sigma=sigma)

    if len(x) > n:
        i_sample = random.sample(list(range(len(x))), n)
        x_sample = x.iloc[i_sample]
        y_sample = y.iloc[i_sample]
    else:
        i_sample = np.array(list(range(len(x))))
        x_sample = x
        y_sample = y

    posx = np.digitize(x_sample, x_center)
    posy = np.digitize(y_sample, y_center)
    z = H_blurred[posx - 1, posy - 1]
    z = z / z.max()

    return x_sample, y_sample, z, i_sample


def get_label(name):
    """Get the label for a given variable.

    :param name: string
        Variable name.

    :return: The label of the variable.
    """
    name = name.lower()

    if name == 'gamma':
        return r"$\gamma$"
    elif name == 'SdE':
        return r"$\sigma_E$"
    elif name == 'Sx':
        return "$\sigma_x$"
    elif name == 'Sy':
        return "$\sigma_y$"
    elif name == 'Sz':
        return "$\sigma_z$"
    elif name == 'betax':
        return r"$\beta_x$"
    elif name == 'betay':
        return r"$\beta_y$"
    elif name == 'alphax':
        return r"$\alpha_x$"
    elif name == 'alphay':
        return r"$\alpha_y$"
    elif name == 'emitx':
        return r"$\varepsilon_x$"
    elif name == 'emity':
        return r"$\varepsilon_y$"
    elif name == 'emitx_tr':
        return r"$\varepsilon_{x, trace}$"
    elif name == 'emity_tr':
        return r"$\varepsilon_{y, trace}$"
    elif name == 'xp':
        return r"$x^\prime$"
    elif name == 'yp':
        return r"$y^\prime$"
    else:
        return r"${}$".format(name)


def get_default_unit(name):
    """Get the default unit of a variable.

    :param name: string
        Variable name.
    """
    name = name.lower()

    if name == 'x' or name == 'y':
        return 'mm'
    elif name == 'z':
        return 'm'
    elif name == 'xp' or name == 'yp':
        return 'mrad'
    elif name == 't':
        return 'fs'
    elif re.match('beta', name):
        return 'm'
    elif name == 'SdE':
        return 'keV'
    elif name == 'Sx' or name == 'Sy':
        return 'mm'
    elif name == 'Sz':
        return 'um'
    elif re.match('emit', name):
        return 'um'
    elif name == 'St':
        return 'fs'
    elif name == 'p':
        return 'mc'
    else:
        return ''


def get_unit_scale(unit):
    """Obtain the label and scaling factor of the unit

    :param unit: string
        Name of the unit.

    :return unit_label: string
        label of the unit
    :return scale: int/float
        Scaling factor of the unit
    """
    unit_label = unit

    if unit == 'MeV':
        scale = 1.0e-6
    elif unit in ['kA', 'keV']:
        scale = 1.0e-3
    elif unit == '' or unit in ['m', 'rad', 's', 'A', 'mc']:
        scale = 1.0
    elif unit in ['mm', 'mrad', 'ms']:
        scale = 1.0e3
    elif unit in ['um', 'urad', 'us']:
        scale = 1.0e6
        if unit == 'um':
            unit_label = '$\mu$m'
        elif unit == 'urad':
            unit_label = '$\mu$rad'
        elif unit == 'us':
            unit_label = '$\mu$s'
    elif unit == 'ps':
        scale = 1.0e12
    elif unit == 'fs':
        scale = 1.0e15
    else:
        raise ValueError("\nUnknown unit!")

    if unit_label:
        unit_label = "(" + unit_label + ")"

    return unit_label, scale
