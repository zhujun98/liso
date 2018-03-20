#!/usr/bin/python
"""
Author: Jun Zhu
"""
import csv

from .phasespace_parser import *


def convert_particle_file(pout, pin, *, code_pout=None, code_pin=None, z0=None, **kwargs):
    """Convert a particle file from one code to another.

    :parm pout: string
        Path of the input particle file.
    :param pin: string
        Path of the output particle file.
    :param code_pout: string
        Code for the input particle file.
    :param code_pin: string
        Code for the output particle file.
    :param z0: float / None
        Beam longitudinal central coordinate for the output particles.
        If None, it equals to the value of the input particles.
    """
    if code_pout is None or code_pin is None:
        raise ValueError("Type for particle file is missing!")

    data, charge = parse_phasespace(code_pout, pout)
    if z0 is not None:
        data['z'] -= (data['z'].mean() - z0)

    if code_pin.lower() in ("astra", "a"):
        raise NotImplementedError
    elif code_pin.lower() in ('impactt', 't'):
        data_to_impactt(data, pin, **kwargs)
    elif code_pin.lower() in ('impactz', 'z'):
        raise NotImplementedError
    elif code_pin.lower() in ('genesis', 'g'):
        raise NotImplementedError
    else:
        raise ValueError("Unknown code!")


def data_to_impactt(data, fpath, *, header=True):
    """Dump the data to an Impact-T particle file.

    :param data: Pandas.DataFrame
        Particle data. See data_processing/phasespace_parser for details
        of the data columns.
    :param fpath: string
        Path of the output particle file.
    :param header: Bool
        True for an input file for Impact-T simulation and False for a
        general particle file.
    """
    with open(fpath, 'w') as fp:
        if header is True:
            fp.write(str(data.shape[0]) + '\n')

        data.to_csv(fp,
                    header=None,
                    index=False,
                    sep=' ',
                    quoting=csv.QUOTE_NONE,
                    escapechar=' ',
                    float_format="%.12E",
                    columns=['x', 'px', 'y', 'py', 'z', 'pz'])
