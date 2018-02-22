#!/usr/bin/python
"""
Author: Jun Zhu

Line data parser for different codes.

The returning data is a pandas.DataFrame containing the following columns:

    z (m)
    gamma
    SdE (eV)
    Sx (m)
    Sy (m)
    Sz (m),
    emitx (m.rad)
    emity (m.rad)
    emitz (m.rad),
    emitx_tr (m.rad)
    emity_tr (m.rad)
    betax (m)
    betay (m)
    alphax
    alphay

"""
import pandas as pd
import numpy as np


from ..backend import config

V_LIGHT = config['vLight']
M_E = config['me']
Q_E = config['qe']

CONST_E = M_E*V_LIGHT**2/Q_E


def parse_impactt_line(root_name):
    """Parse the IMPACT-T line output file.

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
    y_file = root_name + '.25'
    z_file = root_name + '.26'

    xdata = pd.read_csv(
        x_file, delim_whitespace=True,
        names=['t', 'z', 'Cx', 'Sx', 'px', 'Spx', 'x_px', 'emitx'])
    ydata = pd.read_csv(
        y_file, delim_whitespace=True,
        names=['t', 'z', 'Cy', 'Sy', 'py', 'Spy', 'y_py', 'emity'])
    zdata = pd.read_csv(
        z_file, delim_whitespace=True,
        names=['t', 'z', 'Sz', 'pz', 'Spz', 'z_pz', 'emitz'])

    data = pd.DataFrame()

    data['z'] = xdata['z']
    # data['t'] = xdata['t']

    data['gamma'] = np.sqrt(xdata['px'].pow(2) + ydata['py'].pow(2) +
                                 zdata['pz'].pow(2) + 1)
    data['SdE'] = np.sqrt(xdata['Spx'].pow(2) + ydata['Spy'].pow(2) +
                               zdata['Spz'].pow(2))*CONST_E

    data['Sx'] = xdata['Sx']
    data['Sy'] = ydata['Sy']
    data['Sz'] = zdata['Sz']

    data['emitx'] = xdata['emitx']
    data['emity'] = ydata['emity']
    data['emitz'] = zdata['emitz']

    data['emitx_tr'] = xdata['emitx']
    data['emity_tr'] = ydata['emity']
    # data['emitz_tr'] = zdata['emitz']

    gamma_beta = data['gamma']*np.sqrt(1 - 1/data['gamma'].pow(2))

    data['betax'] = data['Sx'].pow(2)*gamma_beta/data['emitx_tr']
    data['betay'] = data['Sy'].pow(2)*gamma_beta/data['emity_tr']

    data['alphax'] = xdata['x_px']/data['emitx_tr']
    data['alphay'] = ydata['y_py']/data['emity_tr']

    return data


def parse_astra_line(root_name):
    """Parse the ASTRA line output file.

    Columns in the input files
    --------------------------
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
    y_file = root_name + '.Yemit.001'
    z_file = root_name + '.Zemit.001'
    emit_tr_file = root_name + '.TRemit.001'

    data = pd.DataFrame()

    xdata = pd.read_csv(
        x_file, delim_whitespace=True,
        names=['z', 't', 'Cx', 'Sx', 'Sxp', 'emitx', 'x_xp'])
    ydata = pd.read_csv(
        y_file, delim_whitespace=True,
        names=['z', 't', 'Cy', 'Sy', 'Syp', 'emity', 'y_yp'])
    zdata = pd.read_csv(
        z_file, delim_whitespace=True,
        names=['z', 't', 'Ek', 'Sz', 'SdE', 'emitz', 'z_dE'])

    # ASTRA will not output .TRemit file by default
    try:
        emit_tr_data = pd.read_csv(
            emit_tr_file, delim_whitespace=True,
            names=['z', 't', 'emitx_tr', 'emity_tr', 'emitz_tr'])
        data['emitx_tr'] = emit_tr_data['emitx_tr']*1.0e-6
        data['emity_tr'] = emit_tr_data['emity_tr']*1.0e-6
        # data['emitz_tr'] = emit_tr_data['emitz_tr']*1.0e-6
    except IOError:
        data['emitx_tr'] = xdata['emitx']*1.0e-6
        data['emity_tr'] = ydata['emity']*1.0e-6
        # data['emitz_tr'] = zdata['emitz']*1.0e-6

    data['z'] = xdata['z']
    # data['t'] = xdata['t']*1.0e-9

    data['gamma'] = zdata['Ek']*1.0e6/CONST_E + 1
    data['SdE'] = zdata['SdE']*1.0e3

    data['Sx'] = xdata['Sx']*1.0e-3
    data['Sy'] = ydata['Sy']*1.0e-3
    data['Sz'] = zdata['Sz']*1.0e-3

    # data['Sxp'] = xdata['Sxp']*1.0e-3
    # data['Syp'] = ydata['Syp']*1.0e-3

    data['emitx'] = xdata['emitx']*1.0e-6
    data['emity'] = ydata['emity']*1.0e-6
    data['emitz'] = zdata['emitz']*1.0e-6

    gamma_beta = data['gamma']*np.sqrt(1 - 1/data['gamma'].pow(2))

    data['betax'] = data['Sx'].pow(2)*gamma_beta/data['emitx_tr']
    data['betay'] = data['Sy'].pow(2)*gamma_beta/data['emity_tr']

    x_xp = xdata['Sx']*xdata['x_xp']*1.0e-6
    y_yp = ydata['Sy']*ydata['y_yp']*1.0e-6
    data['alphax'] = -x_xp*gamma_beta/data['emitx_tr']
    data['alphay'] = -y_yp*gamma_beta/data['emity_tr']

    return data


def parse_impactz_line(rootname):
    raise NotImplemented


def parse_genesis_line(rootname):
    raise NotImplemented
