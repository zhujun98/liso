"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
from abc import abstractmethod
import os
import re

import numpy as np
from scipy import constants

from ..exceptions import LisoRuntimeError

MC2_E = constants.m_e * constants.c**2 / constants.e


class ParticleFileGenerator:
    """Simulation particle file generator."""
    def __init__(self, n, q=1.e-9, *,
                 cathode=True, seed=None,
                 dist_x='uniform', sig_x=1e-3,
                 dist_z='gaussian', sig_z=None, z_ref=0.0,
                 dist_pz='isotropic', ek=0.0):
        """Initialization.

        :param int n: number of particles.
        :param float q: bunch charge.
        :param bool cathode: True for generating particles with a time spread
            rather than a spread in the longitudinal position.
        :param int seed: seed for random generator.
        :param str dist_x: 2D transverse particle distribution.
        :param float sig_x: rms of the transverse particle distribution (in meter)
        :param str dist_z: longitudinal particle distribution name.
        :param float sig_z: rms of the longitudinal particle distribution.
        :param float z_ref: reference z (t if cathode == True) coordinate.
        :param str dist_pz: longitudinal momentum distribution.
        :param float ek: kinetic energy (in eV).
        """
        if not isinstance(n, int) or n < 100:
            raise ValueError("n must be an integer larger than 100!")
        self._n = n

        if seed is not None:
            np.random.seed(seed)

        self._data = self._init_data(n)

        if dist_x == 'uniform':
            rn = np.random.rand(2, n)
            # r = 2 * sigma
            r = 2. * np.sqrt(rn[0]) * sig_x
            phi = 2. * np.pi * rn[1]
            self._data[:, 0] = r * np.cos(phi)
            self._data[:, 2] = r * np.sin(phi)
        else:
            raise ValueError(
                f"Unknown transverse particle distribution: {dist_x}")

        self._z_ref = z_ref
        if dist_z == 'gaussian':
            z_data = sig_z * np.random.randn(n)
        else:
            raise ValueError(
                f"Unknown longitudinal particle distribution: {dist_z}")

        if cathode:
            self._data[:, 6] = z_data  # t
        else:
            self._data[:, 4] = z_data  # z
        self._cathode = cathode

        ek_n = ek / MC2_E
        self._pz_ref = np.sqrt(ek_n ** 2 + 2. * ek_n)
        if dist_pz == 'isotropic':
            rn = np.random.rand(2, n)
            r = self._pz_ref
            theta = 2 * np.pi * rn[0]
            phi = np.arccos(1. - 2. * rn[1]) - np.pi / 2.  # [-pi/2, pi/2]
            self._data[:, 1] = r * np.sin(phi) * np.cos(theta)
            self._data[:, 3] = r * np.sin(phi) * np.sin(theta)
            self._data[:, 5] = r * np.cos(phi)
        else:
            raise ValueError(
                f"Unknown longitudinal momentum distribution: {dist_pz}")

        self._q = q

    @staticmethod
    def _init_data(n):
        """Initialize data array.

        columns: x, px, y, py, z, pz, t
        """
        return np.zeros((n, 7), dtype=np.float64)

    def to_astra(self, filepath):
        """Generate an ASTRA particle file.

        col_names: ['x', 'y', 'z', 'px', 'py', 'pz', 't',
                    'q', 'index', 'flag']
        Units: m, m, m, eV/c, eV/c, eV/c, ns, nC, NA, NA

        :param str filepath: path name of the output file.
        """
        data = np.zeros(self._n,
                        dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
                               ('px', 'f8'), ('py', 'f8'), ('pz', 'f8'),
                               ('t', 'f8'), ('q', 'f8'),
                               ('index', 'i8'), ('flag', 'i8')])

        # x, y, px, py with the first one the reference particle
        data['x'][1:] = self._data[1:, 0]
        data['y'][1:] = self._data[1:, 2]
        data['px'][1:] = self._data[1:, 1] * MC2_E  # /mc -> eV/c
        data['py'][1:] = self._data[1:, 3] * MC2_E  # /mc -> eV/c

        pz_ref = self._data[:, 5].mean() \
            if self._pz_ref is None else self._pz_ref
        data['pz'][0] = pz_ref * MC2_E  # /mc -> eV/c
        data['pz'][1:] = (self._data[1:, 5] - pz_ref) * MC2_E  # /mc -> eV/c

        # z (m) / t (ns)
        t_ref = 0.
        z_ref = 0.
        if self._cathode:
            t_ref = self._data[:, 6].mean() \
                if self._z_ref is None else self._z_ref
        else:
            z_ref = self._data[:, 4].mean() \
                if self._z_ref is None else self._z_ref

        data['z'][0] = z_ref
        data['z'][1:] = self._data[1:, 4] - z_ref

        data['t'][0] = t_ref
        data['t'][1:] = self._data[1:, 6] - t_ref
        data['t'] *= 1.e9  # s -> ns

        # q (in nC)
        data['q'] = -1.e9 * self._q / self._n
        # index (1 for electron)
        data['index'] = 1
        # flag (standard particles)
        data['flag'] = -1 if self._cathode else 5

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w+') as fp:
            np.savetxt(fp, data,
                       fmt=" ".join(["%20.12E"] * 8 + ["%3d"] * 2),
                       delimiter='')

    def to_impactt(self, filepath):
        """Generate an Impact-T particle file.

        col_names: ['x', 'px', 'y', 'py', 'z', 'pz']
        Units: m, /mc, m, /mc, m, /mc

        :param str filepath: path name of the output file.
        """
        data = np.zeros(self._n,
                        dtype=[('x', 'f8'), ('px', 'f8'),
                               ('y', 'f8'), ('py', 'f8'),
                               ('z', 'f8'), ('pz', 'f8')])

        data['x'][:] = self._data[:, 0]
        data['px'][:] = self._data[:, 1]
        data['y'][:] = self._data[:, 2]
        data['py'][:] = self._data[:, 3]
        data['z'][:] = self._data[:, 4]
        data['pz'][:] = self._data[:, 5]

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w+') as fp:
            fp.write(str(self._n) + '\n')
            np.savetxt(fp, data,
                       fmt=" ".join(["%20.12E"] * 6),
                       delimiter='')

    def to_elegant(self, filepath):
        """Generate an Elegant particle file."""
        from sdds import SDDS

        sd = SDDS(0)
        sd.mode = sd.SDDS_BINARY

        sd.description[0] = ""
        sd.description[1] = ""

        sd.parameterName = ["Charge"]
        sd.parameterData = [[self._q]]
        sd.parameterDefinition = [["", "", "", "", sd.SDDS_DOUBLE, ""]]

        xp = self._data[:, 1] / self._data[:, 5]  # px / pz
        yp = self._data[:, 3] / self._data[:, 5]  # py / pz

        z = self._data[:, 4] - self._data[:, 4].mean()
        x = self._data[:, 0] - z * xp
        y = self._data[:, 2] - z * yp

        p = np.sqrt(self._data[:, 1] ** 2
                    + self._data[:, 3] ** 2
                    + self._data[:, 5] ** 2)

        sd.columnName = ["x", "xp", "y", "yp", "p", "t"]
        sd.columnData = [[x.tolist()], [xp.tolist()],
                         [y.tolist()], [yp.tolist()],
                         [p.tolist()], [self._data[:, 6].tolist()]]
        sd.columnDefinition = [["", "m", "", "", sd.SDDS_DOUBLE, 0],
                               ["x'", "", "", "", sd.SDDS_DOUBLE, 0],
                               ["", "m", "", "", sd.SDDS_DOUBLE, 0],
                               ["y'", "", "", "", sd.SDDS_DOUBLE, 0],
                               ["", "m$be$nc", "", "", sd.SDDS_DOUBLE, 0],
                               ["", "s", "", "", sd.SDDS_DOUBLE, 0]]
        sd.save(filepath)
        del sd

    @classmethod
    def from_phasespace(cls, ps):
        """Construct from a Phasespace instance.

        :param Phasespace ps: phasespace.
        """
        instance = cls.__new__(cls)
        super(cls, instance).__init__()

        instance._n = len(ps)
        instance._q = ps.charge

        data = cls._init_data(instance._n)
        data[:, 0] = ps['x']
        data[:, 1] = ps['px']
        data[:, 2] = ps['y']
        data[:, 3] = ps['py']
        data[:, 4] = ps['z']
        data[:, 5] = ps['pz']
        data[:, 6] = ps['t']

        instance._data = data

        instance._cathode = False
        instance._pz_ref = None
        instance._z_ref = None

        return instance


class InputGenerator(object):
    def __init__(self, filepath):
        """Initialization."""
        self._template = self._parse(filepath)
        self._input = None

    @abstractmethod
    def _parse(self, filepath):
        raise NotImplementedError

    def update(self, mapping):
        """Update the input string.

        Patterns in the template input file should be put between
        '<' and '>'.

        :param dict mapping: a pattern-value mapping for replacing the
            pattern with value in the template file.
        """
        found = set()
        self._input = list(self._template)
        for i in range(len(self._input)):
            while True:
                line = self._input[i]

                # Comment line starting with '!'
                if re.match(r'^\s*!', line):
                    break

                left = line.find('<')
                right = line.find('>')
                comment = line.find('!')

                # Cannot find '<' or '>'
                if left < 0 or right < 0:
                    break

                # If '<' is on the right of '>'
                if left >= right:
                    break

                # In line comment
                if left > comment >= 0:
                    break

                ptn = line[left + 1:right]
                try:
                    self._input[i] = line.replace(
                        '<' + ptn + '>', str(mapping[ptn]), 1)
                except KeyError:
                    raise KeyError(
                        "No mapping for <{}> in the template file!".format(ptn))

                found.add(ptn)

        not_found = mapping.keys() - found
        if not_found:
            raise KeyError(f"{not_found} not found in the templates!")

    def write(self, filepath):
        """Write the input string to file.

        :param str filepath: path of the output file.
        """
        if self._input is None:
            raise LisoRuntimeError("Input is not initialized!")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as fp:
            for line in self._input:
                fp.write(line)


class AstraInputGenerator(InputGenerator):
    def _parse(self, filepath):
        """Override."""
        with open(filepath, 'r') as fp:
            template = tuple(fp.readlines())

        return template


class ImpacttInputGenerator(InputGenerator):
    def _parse(self, filepath):
        """Override."""
        with open(filepath, 'r') as fp:
            template = tuple(fp.readlines())

        return template


class ElegantInputGenerator(InputGenerator):
    def _parse(self, filepath):
        """Override."""
        with open(filepath, 'r') as fp:
            template = tuple(fp.readlines())

        return template
