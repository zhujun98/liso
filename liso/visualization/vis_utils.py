"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
# pylint: disable=anomalous-backslash-in-string
import re

from scipy import constants


def get_label(name):  # pylint: disable=too-many-branches,too-many-return-statements
    """Get the label for a given variable.

    :param string name: variable name (case insensitive).

    :return: The label of the variable.
    """
    name = name.lower()

    if name == 'gamma':
        return r"$\gamma$"
    if name == 'sde':
        return r"$\sigma_E$"
    if name == 'delta':
        return r"$\delta$ (%)"
    if name == 'sx':
        return "$\sigma_x$"
    if name == 'sy':
        return "$\sigma_y$"
    if name == 'sz':
        return "$\sigma_z$"
    if name == 'st':
        return "$\sigma_t$"
    if name == 'betax':
        return r"$\beta_x$"
    if name == 'betay':
        return r"$\beta_y$"
    if name == 'alphax':
        return r"$\alpha_x$"
    if name == 'alphay':
        return r"$\alpha_y$"
    if name == 'emitx':
        return r"$\varepsilon_x$"
    if name == 'emity':
        return r"$\varepsilon_y$"
    if name == 'emitx_tr':
        return r"$\varepsilon_{x, trace}$"
    if name == 'emity_tr':
        return r"$\varepsilon_{y, trace}$"
    if name == 'xp':
        return r"$x^\prime$"
    if name == 'yp':
        return r"$y^\prime$"
    if name == 'i':
        return r"$I$"
    return r"${}$".format(name)


def get_html_label(name):  # pylint: disable=too-many-branches,too-many-return-statements
    """Get the label for a given variable.

    :param string name: variable name (case insensitive).

    :return: The label of the variable.
    """
    name = name.lower()

    if name == 'gamma':
        return "<span>&gamma;</span>"
    if name == 'sde':
        return "<span>&sigma;<sub>&delta;</sub></span>"
    if name == 'delta':
        return "<span>&delta; (%)</span>"
    if name == 'sx':
        return "<span>&sigma;<sub>x</sub></span>"
    if name == 'sy':
        return "<span>&sigma;<sub>y</sub></span>"
    if name == 'sz':
        return "<span>&sigma;<sub>z</sub></span>"
    if name == 'st':
        return "<span>&sigma;<sub>t</sub></span>"
    if name == 'betax':
        return "<span>&beta;<sub>x</sub></span>"
    if name == 'betay':
        return "<span>&beta;<sub>y</sub></span>"
    if name == 'alphax':
        return "<span>&alpha;<sub>x</sub></span>"
    if name == 'alphay':
        return "<span>&alpha;<sub>y</sub></span>"
    if name == 'emitx':
        return "<span>&epsilon;<sub>x</sub></span>"
    if name == 'emity':
        return "<span>&epsilon;<sub>y</sub></span>"
    if name == 'emitz':
        return "<span>&epsilon;<sub>z</sub></span>"
    if name == 'emitx_tr':
        return "<span>&epsilon;<sub>x</sub></span>"
    if name == 'emity_tr':
        return "<span>&epsilon;<sub>y</sub></span>"
    if name == 'xp':
        return "<span>x'</span>"
    if name == 'yp':
        return "<span>y'</span>"
    return "<span>{}</span>".format(name)


def get_default_unit(name):  # pylint: disable=too-many-branches,too-many-return-statements
    """Get the default unit of a variable.

    :param string name: variable name (case insensitive).
    """
    name = name.lower()

    if name in ('x', 'y', 'dz'):
        return 'mm'
    if name == 'z':
        return 'm'
    if name in ('xp', 'yp'):
        return 'mrad'
    if name in ('t', 'dt'):
        return 'fs'
    if re.match('beta', name):
        return 'm'
    if name == 'sde':
        return 'kev'
    if name in ('sx', 'sy'):
        return 'mm'
    if name == 'sz':
        return 'um'
    if re.match('emit', name):
        return 'um'
    if name == 'st':
        return 'fs'
    if name == 'p':
        return 'mc'
    if name == 'i':
        return 'a'  # Amper for current I
    return ''


def get_unit_label_and_scale(unit):  # pylint: disable=too-many-branches
    """Obtain the label and scaling factor of the unit

    :param string unit: name of the unit (case insensitive).

    :return string unit_label: label of the unit
    :return float scale: scaling factor of the unit
    """
    unit = unit.lower()

    if unit == 'gev':
        scale = 1.0e-9
        unit = 'GeV'
    elif unit == 'mev':
        scale = 1.0e-6
        unit = 'MeV'
    elif unit == 'kev':
        scale = 1.0e-3
        unit = 'KeV'
    elif unit == 'ka':
        scale = 1.0e-3
        unit = 'kA'
    elif unit == 'a':
        scale = 1.0
        unit = 'A'
    elif unit == '' or unit in ['m', 'rad', 's', 'mc']:
        scale = 1.0
    elif unit in ['mm', 'mrad', 'ms']:
        scale = 1.0e3
    elif unit in ['um', 'urad', 'us']:
        scale = 1.0e6
        if unit == 'um':
            unit = '$\mu$m'
        elif unit == 'urad':
            unit = '$\mu$rad'
        elif unit == 'us':
            unit = '$\mu$s'
    elif unit in ['nm', 'nrad', 'ns']:
        scale = 1.0e9
    elif unit == 'ps':
        scale = 1.0e12
    elif unit == 'fs':
        scale = 1.0e15
    else:
        raise ValueError("\nUnknown unit!")

    if unit:
        unit = "(" + unit + ")"

    return unit, scale


def get_line_column_by_name(data, name):
    """Get the Line column data by name.

    :param string name: name of the column data.
    """
    name = name.lower()

    if name == 'st':
        return data['sz'] / constants.c

    return data[name]
