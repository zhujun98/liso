"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""

# Line data parser for different codes.
#
# The returning data is a pandas.DataFrame containing the following columns:
#
#     x (m)
#     px (mc)
#     y (m)
#     py (mc)
#     z(m)
#     pz (mc)
#     t (s).

from scipy import constants
import numpy as np
import pandas as pd

from .data_proc_utils import check_data_file

V_LIGHT = constants.c
MC2_E = constants.m_e * constants.c**2 / constants.e


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
    data = data[data['flag'].isin([3, 5])]

    p = np.sqrt(data['px'] ** 2 + data['py'] ** 2 + data['pz'] ** 2)

    # At this step, the timing can be used for timing parameter_scan study.
    t_ref = data['t'].iloc[0]/1.e9
    data['t'] = t_ref - (data['z'] - z_ref)/(V_LIGHT * data['pz']/np.sqrt(p ** 2 + 1))
    charge = -1e-9 * data['charge'].sum()

    data.drop(['charge', 'index', 'flag'], inplace=True, axis=1)

    return data, charge


def parse_impactt_phasespace(particle_file):
    """Parse the IMPACT-T particle file.

    :param particle_file: string
        Pathname of the particle file.

    :return data: pandas.DataFrame
        Phase-space data.
    :return charge: None
        Impact-T particle file does not contain charge information.
    """
    # Units: m, /mc, m, /mc, m, /mc
    col_names = ['x', 'px', 'y', 'py', 'z', 'pz']

    data = pd.read_csv(particle_file, delim_whitespace=True, names=col_names)

    # Drop the first row if the input file is 'partcl.data'.
    data.dropna(inplace=True)

    p = np.sqrt(data['px'] ** 2 + data['py'] ** 2 + data['pz'] ** 2)

    # Impact-T does not support timing, here 't' is the relative number
    data['t'] = (data['z'].mean() - data['z']) / (V_LIGHT * data['pz']/np.sqrt(p ** 2 + 1))

    return data, None
