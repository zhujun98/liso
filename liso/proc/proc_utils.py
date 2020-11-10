"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import os

from scipy import constants

from ..exceptions import LisoRuntimeError

MC_E = constants.m_e * constants.c / constants.e


def check_data_file(filepath):
    """Check the status of a data file."""
    if not os.path.isfile(filepath):
        raise LisoRuntimeError(filepath + " does not exist!")
    if not os.path.getsize(filepath):
        raise LisoRuntimeError(filepath + " is empty!")


def quad_k2g(k, p):
    """Convert the K value of a quadrupole to gradient.

    :param k: float
        Quadrupole strength (1/m^2)
    :param p: float
        Normalized momentum

    :returns Quadrupole gradient (T/m)
    """
    return -1.0 * k * p * MC_E
