"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
from scipy import constants
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

from .phasespace import Phasespace
from .proc_utils import check_data_file


V_LIGHT = constants.c
MC2_E = constants.m_e * constants.c**2 / constants.e

# Note:
#      ASTRA: absolute z, absolute t
#   IMPACT-T: absolute z, relative t
#    ELEGANT: relative z, absolute t


def parse_astra_phasespace(particle_file, *, cathode=False):
    """Parse the ASTRA particle file.

    :param string particle_file: pathname of the particle file.
    :param bool cathode: True for a particle file for the cathode.

    :return pandas.DataFrame data: phase-space data.
    :return float charge: charge (C) of the bunch.
    """
    # Units: m, m, m, eV/c, eV/c, eV/c, ns, nC, NA, NA
    col_names = ['x', 'y', 'z', 'px', 'py', 'pz',
                 't', 'charge', 'index', 'flag']

    check_data_file(particle_file)

    data = pd.read_csv(particle_file, delim_whitespace=True, names=col_names)

    pz_ref = data['pz'].iloc[0]
    data.loc[0, 'pz'] = 0.0
    data['pz'] += pz_ref

    data['px'] /= MC2_E
    data['py'] /= MC2_E
    data['pz'] /= MC2_E

    # ix will first try to act like loc to find the index label.
    # If the index label is not found, it will add an index label
    # (new row). Therefore, the reference particle must be used
    # before removing lost particles since the reference particle
    # could be removed.
    z_ref = data['z'].iloc[0]
    data.loc[0, 'z'] = 0.0
    data['z'] += z_ref

    # remove lost particles
    #   -1: standard particle at the cathode (not yet started)
    #   -3: trajectory probe particle at the cathode
    #    3: trajectory probe particle
    #    5: standard particle
    flags = (-1, -3) if cathode else (3, 5)
    data = data[data['flag'].isin(flags)]

    p = np.sqrt(data['px'] ** 2 + data['py'] ** 2 + data['pz'] ** 2)

    # At this step, the timing can be used for timing parameter_scan study.
    t_ref = data['t'].iloc[0] * 1.e-9
    if not cathode:
        data['t'] = t_ref - (data['z'] - z_ref) \
                    / (V_LIGHT * data['pz'] / np.sqrt(p ** 2 + 1))
    else:
        data['t'][1:] = data['t'][1:] * 1.e-9 + t_ref

    charge = -1e-9 * data['charge'].sum()

    data.drop(['charge', 'index', 'flag'], inplace=True, axis=1)

    return Phasespace(data, charge)


def parse_impactt_phasespace(particle_file):
    """Parse the IMPACT-T particle file.

    :param string particle_file: pathname of the particle file.

    :return pandas.DataFrame data: phase-space data.
    :return None charge: Impact-T particle file does not contain charge
        information.
    """
    # Units: m, /mc, m, /mc, m, /mc
    col_names = ['x', 'px', 'y', 'py', 'z', 'pz']

    data = pd.read_csv(particle_file, delim_whitespace=True, names=col_names)

    # Drop the first row if the input file is 'partcl.data'.
    data.dropna(inplace=True)

    p = np.sqrt(data['px'] ** 2 + data['py'] ** 2 + data['pz'] ** 2)

    # Impact-T does not support timing, here 't' is the relative number
    data['t'] = (data['z'].mean() - data['z']) / \
                (V_LIGHT * data['pz'] / np.sqrt(p ** 2 + 1))

    return Phasespace(data, None)


def parse_elegant_phasespace(particle_file):
    from sdds import SDDS

    sd = SDDS(0)
    sd.load(particle_file)

    charge = sd.parameterData[sd.parameterName.index('Charge')][0]

    columns = ['x', 'y', 'xp', 'yp', 'p', 't']
    data = dict()
    for col in columns:
        data[col] = sd.columnData[sd.columnName.index(col)][0]
    data = pd.DataFrame.from_dict(data)
    data['z'] = np.zeros_like(data['t'])

    data['pz'] = data['p'] / np.sqrt(data['xp'] ** 2 + data['yp'] ** 2 + 1)
    data['px'] = data['pz'] * data['xp']
    data['py'] = data['pz'] * data['yp']
    data['dt'] = data['t'] - data['t'].mean()
    data['z'] = data['dt'] * V_LIGHT * data['pz'] / np.sqrt(data['p'] ** 2 + 1)
    data['x'] += data['xp'] * data['z']
    data['y'] += data['yp'] * data['z']

    data.drop(['xp', 'yp', 'p', 'dt'], inplace=True, axis=1)

    return Phasespace(data, charge)
