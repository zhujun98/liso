"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
from ..config import Config

V_LIGHT = Config.vLight
M_E = Config.me
Q_E = Config.qe


def get_code(text):
    """Return the standard code representation letter.

    :param text: string
        Input string.
    """
    if text.lower() in ('a', 'astra'):
        return 'a'
    if text.lower() in ('t', 'impactt'):
        return 't'
    if text.lower() in ('z', 'impactz'):
        return 'z'
    if text.lower() in ('z', 'genesis'):
        return 'g'

    raise ValueError("Unknown code option!")


def quad_k2g(k, p):
    """Convert the K value of a quadrupole to gradient.

    :param k: float
        Quadrupole strength (1/m^2)
    :param p: float
        Normalized momentum

    :returns Quadrupole gradient (T/m)
    """
    return -1.0*k*p*M_E*V_LIGHT/Q_E