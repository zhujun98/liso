"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
from .line_parameters import LineParameters
from ..exceptions import LisoRuntimeError


def analyze_line(data, func, *, min_particles=5):
    """Calculate line parameters.

    :param Pandas.DataFrame data: line data.
    :param callable func: range of the z coordinate.
    :param int min_particles: minimum number of particles required
        for line analysis.

    :return: A LineParameters instance.
    """
    if len(data) < min_particles:
        raise LisoRuntimeError(f"Too few points {len(data)} in the line")

    params = LineParameters()

    params.z = func(data['z'])
    params.gamma = func(data['gamma'])
    params.SdE = func(data['sde'])
    params.Sx = func(data['sx'])
    params.Sy = func(data['sy'])
    params.Sz = func(data['sz'])
    params.betax = func(data['betax'])
    params.betay = func(data['betay'])
    params.alphax = func(data['alphax'])
    params.alphay = func(data['alphay'])
    params.emitx = func(data['emitx'])
    params.emity = func(data['emity'])
    params.emitx_tr = func(data['emitx_tr'])
    params.emity_tr = func(data['emity_tr'])

    return params
