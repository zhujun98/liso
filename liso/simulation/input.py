"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
from abc import abstractmethod
import csv

import numpy as np
from scipy import constants

MC2_E = constants.m_e * constants.c**2 / constants.e


class ParticleFileGenerator:
    """Simulation particle file generator."""
    def __init__(self, n, q=1.e-9, *,
                 cathode=True, seed=None,
                 dist_x='uniform', sig_x=1e-3,
                 dist_z='gaussian', sig_z=1e-12, ref_z=0.0,
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
        :param float ref_z: reference z coordinate.
        :param str dist_pz: longitudinal momentum distribution.
        :param float ek: kinetic energy (in eV).
        """
        if not isinstance(n, int) or n < 100:
            raise ValueError("n must be an integer larger than 100!")
        self._n = n

        self._cathode = cathode

        if seed is not None:
            np.random.seed(seed)

        self._data = np.zeros((n, 6), dtype=np.float64)

        if dist_x == 'uniform':
            data = np.random.rand(2, n)
            # r = 2 * sigma
            r = 2. * np.sqrt(data[0]) * sig_x
            phi = 2. * np.pi * data[1]
            self._data[:, 0] = r * np.cos(phi)
            self._data[:, 1] = r * np.sin(phi)
        else:
            raise ValueError(
                f"Unknown transverse particle distribution: {dist_x}")

        self._ref_z = ref_z
        if dist_z == 'gaussian':
            self._data[:, 2] = sig_z * np.random.randn(n)
        else:
            raise ValueError(
                f"Unknown longitudinal particle distribution: {dist_z}")

        ek_n = ek / MC2_E
        self._p = np.sqrt(ek_n ** 2 + 2. * ek_n)
        if dist_pz == 'isotropic':
            data = np.random.rand(2, n)
            r = self._p
            theta = 2 * np.pi * data[0]
            phi = np.arccos(1. - 2. * data[1]) - np.pi / 2.  # [-pi/2, pi/2]
            self._data[:, 3] = r * np.sin(phi) * np.cos(theta)
            self._data[:, 4] = r * np.sin(phi) * np.sin(theta)
            self._data[:, 5] = r * np.cos(phi)
        else:
            raise ValueError(
                f"Unknown longitudinal momentum distribution: {dist_pz}")

        self._q = q

    def toAstra(self, filepath):
        """Generate an ASTRA particle file.

        col_names = ['x', 'y', 'z', 'px', 'py', 'pz', 't',
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
        data['y'][1:] = self._data[1:, 1]
        data['px'][1:] = self._data[1:, 3] * MC2_E  # /mc -> eV/c
        data['py'][1:] = self._data[1:, 4] * MC2_E  # /mc -> eV/c

        data['pz'][0] = self._p * MC2_E  # /mc -> eV/c
        data['pz'][1:] = (self._data[1:, 5] - self._p) * MC2_E  # /mc -> eV/c

        # t (ns)
        if self._cathode:
            data['t'][0] = self._ref_z
            data['t'][1:] = self._data[1:, 2] - self._ref_z
            data['t'] *= 1.e9  # s -> ns
        else:
            data['z'][0] = self._ref_z
            data['z'][1:] = self._data[1:, 2] - self._ref_z

        # q (in nC)
        data['q'] = -1.e9 * self._q / self._n
        # index (1 for electron)
        data['index'] = 1
        # flag (-1 for standard particle)
        data['flag'] = -1

        with open(filepath, 'w') as fp:
            np.savetxt(fp, data,
                       fmt=" ".join(["%20.12E"] * 8 + ["%3d"] * 2),
                       delimiter='')

    @classmethod
    def fromDataframeToAstra(cls, data, filepath):
        """Generate an Astra particle file from a data frame.

        :param pandas.DataFrame data: data frame.
        :param str filepath: path name of the output file.
        """
        raise NotImplementedError

    @classmethod
    def fromDataframeToImpactt(cls, data, filepath):
        """Generate an Impact-T particle file from a data frame.

        :param pandas.DataFrame data: data frame.
        :param str filepath: path name of the output file.
        """
        with open(filepath, 'w') as fp:
            fp.write(str(data.shape[0]) + '\n')
            data.to_csv(fp,
                        header=False,
                        index=False,
                        sep=' ',
                        quoting=csv.QUOTE_NONE,
                        escapechar=' ',
                        float_format="%.12E",
                        columns=['x', 'px', 'y', 'py', 'z', 'pz'])


class InputGenerator(object):
    def __init__(self):
        """Initialization."""

    @abstractmethod
    def add_quad(self):
        raise NotImplemented

    @abstractmethod
    def add_dipole(self):
        raise NotImplemented

    @abstractmethod
    def add_tws(self):
        raise NotImplemented

    @abstractmethod
    def add_gun(self):
        raise NotImplemented


class AstraInputGenerator(InputGenerator):
    pass


class ImpacttInputGenerator(InputGenerator):
    pass
