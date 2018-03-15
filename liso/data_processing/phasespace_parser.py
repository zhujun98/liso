#!/usr/bin/python
"""
Author: Jun Zhu

Line data parser for different codes.

The returning data is a pandas.DataFrame containing the following columns:

    x (m)
    px (mc)
    y (m)
    py (mc)
    z(m)
    pz (mc)
    t (s).
"""
import numpy as np
import pandas as pd

from ..backend import config


V_LIGHT = config['vLight']
M_E = config['me']
Q_E = config['qe']

CONST_E = M_E*V_LIGHT**2/Q_E


def parse_phasespace(code, particle_file):
    """Parse a particle file from different code.

    :param code: string
        Name of the code.
    :param particle_file: string
        Pathname of the particle file.

    :return data: pandas.dataframe
        Data for each particle.
    :return charge: float / None
        Charge (C) of the bunch.
    """
    if code.lower() in ("astra", "a"):
        data, charge = parse_astra_phasespace(particle_file)
    elif code.lower() in ('impactt', 't'):
        data = parse_impactt_phasespace(particle_file)
        charge = None
    elif code.lower() in ('impactz', 'z'):
        data = parse_impactz_phasespace(particle_file)
        charge = None
    elif code.lower() in ('genesis', 'g'):
        raise NotImplementedError
    else:
        raise ValueError("Unknown code!")

    return data, charge


def parse_astra_phasespace(particle_file):
    """Parse the ASTRA particle file.

    :param particle_file: string
        Pathname of the particle file.

    :return data: pandas.DataFrame
        Phase-space data.
    :return charge: float
        Charge (C) of the bunch.
    """
    # Units: m, m, m, eV/c, eV/c, eV/c, ns, nC, NA, NA
    col_names = ['x', 'y', 'z', 'px', 'py', 'pz',
                 't', 'charge', 'index', 'flag']

    data = pd.read_csv(particle_file, delim_whitespace=True, names=col_names)

    pz_ref = data['pz'].iloc[0]
    data.ix[0, 'pz'] = 0.0
    data['pz'] += pz_ref

    tmp = M_E * V_LIGHT ** 2 / Q_E
    data['px'] /= tmp
    data['py'] /= tmp
    data['pz'] /= tmp

    # ix will first try to act like loc to find the index label.
    # If the index label is not found, it will add an index label
    # (new row). Therefore, the reference particle must be used
    # before removing lost particles since the reference particle
    # could be removed.
    z_ref = data['z'].iloc[0]
    data.ix[0, 'z'] = 0.0
    data['z'] += z_ref

    # remove lost particles
    data = data[data['flag'].isin([3, 5])]

    p = np.sqrt(data['px'] ** 2 + data['py'] ** 2 + data['pz'] ** 2)

    # At this step, the timing can be used for timing jitter study.
    data['t'] = data['t'].iloc[0]/1e9 - (data['z'] - z_ref)\
        / (V_LIGHT * data['pz']/np.sqrt(p ** 2 + 1))

    # The bunch is centered for the convenience of the longitudinal
    # phase-space plot.
    data['t'] = data['t'] - data['t'].mean()

    charge = -1e-9 * data['charge'].sum()

    data.drop(['charge', 'index', 'flag'], inplace=True, axis=1)

    return data, charge


def parse_impactt_phasespace(particle_file):
    """Parse the IMPACT-T particle file.

    :param particle_file: string
        Pathname of the particle file.

    :return data: pandas.DataFrame
        Phase-space data.
    """
    # Units: m, /mc, m, /mc, m, /mc
    col_names = ['x', 'px', 'y', 'py', 'z', 'pz']

    data = pd.read_csv(particle_file, delim_whitespace=True, names=col_names)

    # Drop the first row if the input file is 'partcl.data'.
    data.dropna(inplace=True)

    p = np.sqrt(data['px'] ** 2 + data['py'] ** 2 + data['pz'] ** 2)

    data['t'] = -(data['z'] - data['z'].mean()) \
        / (V_LIGHT * data['pz'] / np.sqrt(p ** 2 + 1))

    return data


def parse_impactz_phasespace(particle_file):
    raise NotImplemented


def parse_genesis_phasespace(particle_file):
    raise NotImplemented
