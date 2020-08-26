"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import re

from scipy import constants


def get_label(name):
    """Get the label for a given variable.

    :param string name: variable name (case insensitive).

    :return: The label of the variable.
    """
    name = name.lower()

    if name == 'gamma':
        return r"$\gamma$"
    elif name == 'sde':
        return r"$\sigma_E$"
    elif name == 'delta':
        return r"$\delta$ (%)"
    elif name == 'sx':
        return "$\sigma_x$"
    elif name == 'sy':
        return "$\sigma_y$"
    elif name == 'sz':
        return "$\sigma_z$"
    elif name == 'st':
        return "$\sigma_t$"
    elif name == 'betax':
        return r"$\beta_x$"
    elif name == 'betay':
        return r"$\beta_y$"
    elif name == 'alphax':
        return r"$\alpha_x$"
    elif name == 'alphay':
        return r"$\alpha_y$"
    elif name == 'emitx':
        return r"$\varepsilon_x$"
    elif name == 'emity':
        return r"$\varepsilon_y$"
    elif name == 'emitx_tr':
        return r"$\varepsilon_{x, trace}$"
    elif name == 'emity_tr':
        return r"$\varepsilon_{y, trace}$"
    elif name == 'xp':
        return r"$x^\prime$"
    elif name == 'yp':
        return r"$y^\prime$"
    elif name == 'i':
        return r"$I$"
    else:
        return r"${}$".format(name)


def get_html_label(name):
    """Get the label for a given variable.

    :param string name: variable name (case insensitive).

    :return: The label of the variable.
    """
    name = name.lower()

    if name == 'gamma':
        return "<span>&gamma;</span>"
    elif name == 'sde':
        return "<span>&sigma;<sub>&delta;</sub></span>"
    elif name == 'delta':
        return "<span>&delta; (%)</span>"
    elif name == 'sx':
        return "<span>&sigma;<sub>x</sub></span>"
    elif name == 'sy':
        return "<span>&sigma;<sub>y</sub></span>"
    elif name == 'sz':
        return "<span>&sigma;<sub>z</sub></span>"
    elif name == 'st':
        return "<span>&sigma;<sub>t</sub></span>"
    elif name == 'betax':
        return "<span>&beta;<sub>x</sub></span>"
    elif name == 'betay':
        return "<span>&beta;<sub>y</sub></span>"
    elif name == 'alphax':
        return "<span>&alpha;<sub>x</sub></span>"
    elif name == 'alphay':
        return "<span>&alpha;<sub>y</sub></span>"
    elif name == 'emitx':
        return "<span>&epsilon;<sub>x</sub></span>"
    elif name == 'emity':
        return "<span>&epsilon;<sub>y</sub></span>"
    elif name == 'emitz':
        return "<span>&epsilon;<sub>z</sub></span>"
    elif name == 'emitx_tr':
        return "<span>&epsilon;<sub>x</sub></span>"
    elif name == 'emity_tr':
        return "<span>&epsilon;<sub>y</sub></span>"
    elif name == 'xp':
        return "<span>x'</span>"
    elif name == 'yp':
        return "<span>y'</span>"
    else:
        return "<span>{}</span>".format(name)


def get_default_unit(name):
    """Get the default unit of a variable.

    :param string name: variable name (case insensitive).
    """
    name = name.lower()

    if name == 'x' or name == 'y' or name == 'dz':
        return 'mm'
    elif name == 'z':
        return 'm'
    elif name == 'xp' or name == 'yp':
        return 'mrad'
    elif name == 't' or name == 'dt':
        return 'fs'
    elif re.match('beta', name):
        return 'm'
    elif name == 'sde':
        return 'kev'
    elif name == 'sx' or name == 'sy':
        return 'mm'
    elif name == 'sz':
        return 'um'
    elif re.match('emit', name):
        return 'um'
    elif name == 'st':
        return 'fs'
    elif name == 'p':
        return 'mc'
    elif name == 'i':
        return 'a'  # Amper for current I
    else:
        return ''


def get_unit_label_and_scale(unit):
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
