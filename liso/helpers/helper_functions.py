#!/usr/bin/python
"""
Author: Jun Zhu

"""
from ..backend import config

V_LIGHT = config['V_LIGHT']
M_E = config['M_E']
Q_E = config['Q_E']


def quad_k2g(k, p):
    """Convert the K value of a quadrupole to gradient.

    :param k: float
        Quadrupole strength (1/m^2)
    :param p: float
        Normalized momentum

    :returns Quadrupole gradient (T/m)
    """
    return -1.0*k*p*M_E*V_LIGHT/Q_E