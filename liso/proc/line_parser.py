"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""

# Line data parser for different codes.
#
# Important note:
# _______________
# The calculated Twiss parameters are based on the canonical coordinates
# of the beam. Therefore, when the beam has a large energy spread or a
# big divergence angle, the result will be quite different from the
# true Twiss parameters. However, one can set TR_EmitS = .T in ASTRA to
# get the very-close Twiss parameters even in extreme conditions.
#
#
# The data is a pandas.DataFrame containing the following columns:
#
#     z (m)
#     gamma
#     SdE (eV)
#     Sx (m)
#     Sy (m)
#     Sz (m),
#     emitx (m.rad)
#     emity (m.rad)
#     emitz (m.rad),
#     emitx_tr (m.rad)
#     emity_tr (m.rad)
#     betax (m)
#     betay (m)
#     alphax
#     alphay

from scipy import constants
import pandas as pd
import numpy as np

from .proc_utils import check_data_file
from ..exceptions import LisoRuntimeError

MC2_E = constants.m_e * constants.c**2 / constants.e


def parse_impactt_line(root_name):
    """Parse the IMPACT-T output file.

    Columns in the input files:

    xdata:
        t (s), z (m), Cx (m), Sx (m), px (/mc), Spx (/mc),
        twiss (m), emitx (m).
    ydata:
        t (s), z (m), Cy (m), Sy (m), py (/mc), Spy (/mc),
        twiss (m), emity (m).
    zdata:.
        t (s), z (m), Sz (m), pz (/mc), Spz (/mc),
        twiss (m), emitz (m).

    Note: twiss = -<x - <x>><px - <px>>
    """
    if root_name is None:
        root_name = 'fort'

    x_file = root_name + '.24'
    check_data_file(x_file)
    y_file = root_name + '.25'
    check_data_file(y_file)
    z_file = root_name + '.26'
    check_data_file(z_file)

    xdata = pd.read_csv(
        x_file, delim_whitespace=True,
        names=['t', 'z', 'cx', 'sx', 'px', 'spx', 'x_px', 'emitx'])
    ydata = pd.read_csv(
        y_file, delim_whitespace=True,
        names=['t', 'z', 'cy', 'sy', 'py', 'spy', 'y_py', 'emity'])
    zdata = pd.read_csv(
        z_file, delim_whitespace=True,
        names=['t', 'z', 'sz', 'pz', 'spz', 'z_pz', 'emitz'])

    data = pd.DataFrame()

    data['z'] = xdata['z']
    # data['t'] = xdata['t']

    data['gamma'] = np.sqrt(xdata['px'].pow(2) + ydata['py'].pow(2) +
                                 zdata['pz'].pow(2) + 1)
    data['sde'] = np.sqrt(xdata['spx'].pow(2) + ydata['spy'].pow(2) +
                               zdata['spz'].pow(2)) * MC2_E

    data['sx'] = xdata['sx']
    data['sy'] = ydata['sy']
    data['sz'] = zdata['sz']

    data['emitx'] = xdata['emitx']
    data['emity'] = ydata['emity']
    data['emitz'] = zdata['emitz']

    data['emitx_tr'] = xdata['emitx']
    data['emity_tr'] = ydata['emity']
    # data['emitz_tr'] = zdata['emitz']

    gamma_beta = data['gamma']*np.sqrt(1 - 1/data['gamma'].pow(2))

    data['betax'] = data['sx'].pow(2)*gamma_beta/data['emitx_tr']
    data['betay'] = data['sy'].pow(2)*gamma_beta/data['emity_tr']

    data['alphax'] = xdata['x_px']/data['emitx_tr']
    data['alphay'] = ydata['y_py']/data['emity_tr']

    return data


def parse_astra_line(root_name):
    """Parse the ASTRA output file.

    Columns in the input files:

    xdata:
        z (m), t (ns), Cx (mm), Sx (mm), Sxp (mm), emitx (um),
        x_xp (um).
    ydata:
        z (m), t (ns), Cy (mm), Sy (mm), Syp (mm), emity (um),
        y_yp (um).
    zdata:
        z (m), t (ns), Ek (MeV), Sz (mm), SdE (keV), emitz (um),
        z_dE (um).

    Note: x_xp = <x.*xp>/sqrt(<x^2>)/<pz>,
          y_yp = <y.*yp>/sqrt(<y^2>)/<pz>,
          z_de = <z.*dE>/sqrt(<z^2>),
          Sxp = Spx/<pz>, this is not Sxp, so I cannot calculate
          the trace-space emittance!!!
          Syp = Spy/<pz>.

          Even thought the trace-space emittance is read from the
          TRemit file, there is still a little difference between
          the correct value since we do not know the real <x.*xp>
    """
    if root_name is None:
        raise ValueError("\nroot_name of the output files is not given!")

    x_file = root_name + '.Xemit.001'
    check_data_file(x_file)
    y_file = root_name + '.Yemit.001'
    check_data_file(y_file)
    z_file = root_name + '.Zemit.001'
    check_data_file(z_file)
    emit_tr_file = root_name + '.TRemit.001'

    data = pd.DataFrame()

    xdata = pd.read_csv(
        x_file,
        delim_whitespace=True,
        names=['z', 't', 'cx', 'sx', 'sxp', 'emitx', 'x_xp'])
    ydata = pd.read_csv(
        y_file,
        delim_whitespace=True,
        names=['z', 't', 'cy', 'sy', 'syp', 'emity', 'y_yp'])
    zdata = pd.read_csv(
        z_file,
        delim_whitespace=True,
        names=['z', 't', 'ek', 'sz', 'sde', 'emitz', 'z_de'])

    # ASTRA will not output .TRemit file by default
    try:
        check_data_file(emit_tr_file)
    
        emit_tr_data = pd.read_csv(
            emit_tr_file, delim_whitespace=True,
            names=['z', 't', 'emitx_tr', 'emity_tr', 'emitz_tr'])
        data['emitx_tr'] = emit_tr_data['emitx_tr']*1.0e-6
        data['emity_tr'] = emit_tr_data['emity_tr']*1.0e-6
        # data['emitz_tr'] = emit_tr_data['emitz_tr']*1.0e-6
    except LisoRuntimeError:
        data['emitx_tr'] = xdata['emitx']*1.0e-6
        data['emity_tr'] = ydata['emity']*1.0e-6
        # data['emitz_tr'] = zdata['emitz']*1.0e-6

    data['z'] = xdata['z']
    # data['t'] = xdata['t']*1.0e-9

    data['gamma'] = zdata['ek']*1.0e6 / MC2_E + 1
    data['sde'] = zdata['sde']*1.0e3

    data['sx'] = xdata['sx']*1.0e-3
    data['sy'] = ydata['sy']*1.0e-3
    data['sz'] = zdata['sz']*1.0e-3

    # data['sxp'] = xdata['sxp']*1.0e-3
    # data['syp'] = ydata['syp']*1.0e-3

    data['emitx'] = xdata['emitx']*1.0e-6
    data['emity'] = ydata['emity']*1.0e-6
    data['emitz'] = zdata['emitz']*1.0e-6

    gamma_beta = data['gamma']*np.sqrt(1 - 1/data['gamma'].pow(2))

    data['betax'] = data['sx'].pow(2)*gamma_beta/data['emitx_tr']
    data['betay'] = data['sy'].pow(2)*gamma_beta/data['emity_tr']

    x_xp = xdata['sx']*xdata['x_xp']*1.0e-6
    y_yp = ydata['sy']*ydata['y_yp']*1.0e-6
    data['alphax'] = -x_xp*gamma_beta/data['emitx_tr']
    data['alphay'] = -y_yp*gamma_beta/data['emity_tr']

    return data


def parse_elegant_line(root_name):
    pass
