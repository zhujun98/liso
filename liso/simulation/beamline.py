"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import asyncio
import os.path as osp
from abc import ABC, abstractmethod
import subprocess
from distutils.spawn import find_executable

import numpy as np

from ..config import config
from ..data_processing import (
    analyze_beam, analyze_line, ParticleFileGenerator,
    parse_astra_phasespace, parse_impactt_phasespace,
    parse_astra_line, parse_impactt_line,
)
from .simulation_utils import generate_input


class Beamline(ABC):
    """Beamline abstraction class."""

    def __init__(self, name, *,
                 template=None,
                 fin=None,
                 pout=None,
                 charge=None,
                 z0=None):
        """Initialization.

        :param str name: Name of the beamline.
        :param str template: path of the template of the input file.
        :param str fin: path of the input file. The simulation working directory
            is assumed to be the directory of the input file.
        :param str pout: final particle file name. It must be located in the
            simulation working directory.
        :param float charge: Bunch charge at the beginning of the beamline.
            Only used for certain codes (e.g. ImpactT).
        :param float z0: Starting z coordinate in meter. Used for concatenated
            simulation. Default = None, inherit z coordinate from the upstream
            beamline. However, for instance, when the second beamline is
            defined from z0 = 0.0, z0 is required to generate a correct initial
            particle distribution.
        """
        self.name = name

        # Read the template file only once. Make the 'template' read-only by
        # converting it to a tuple.
        with open(template) as fp:
            self.template = tuple(fp.readlines())

        self._swd = osp.dirname(osp.abspath(fin))
        self._fin = osp.join(self._swd, osp.basename(fin))

        # Initial particle distribution file name.
        self._pin = None
        self._pout = osp.join(self._swd, pout)
        self._rootname = None

        self._charge = charge

        self.z0 = z0  # starting z coordinate (m)

        # BeamParameters
        self._out = None

        # LineParameters
        self._start = None
        self._end = None
        self._min = None
        self._max = None
        self._avg = None
        self._std = None

        # suffixes for the output files related to Line instance.
        self._output_suffixes = []

    @property
    def out(self):
        if self._out is None:
            data, charge = self._parse_phasespace(self._pout)
            charge = self._charge if charge is None else charge
            self._out = analyze_beam(data, charge)
        return self._out

    @property
    def start(self):
        if self._start is None:
            self._update_statistics()
        return self._start

    @property
    def end(self):
        if self._end is None:
            self._update_statistics()
        return self._end

    @property
    def min(self):
        if self._min is None:
            self._update_statistics()
        return self._min

    @property
    def max(self):
        if self._max is None:
            self._update_statistics()
        return self._max

    @property
    def avg(self):
        if self._avg is None:
            self._update_statistics()
        return self._avg

    @property
    def std(self):
        if self._std is None:
            self._update_statistics()
        return self._std

    def _update_statistics(self):
        data = self._parse_line()

        self._start = analyze_line(data, lambda x: x.iloc[0])
        self._end = analyze_line(data, lambda x: x.iloc[-1])
        self._min = analyze_line(data, np.min)
        self._max = analyze_line(data, np.max)
        self._avg = analyze_line(data, np.average)
        self._std = analyze_line(data, np.std)

    @abstractmethod
    def generate_initial_particle_file(self, data, charge):
        """Generate the initial particle file.

        :param data: Pandas.DataFrame
            Particle data. See data_processing/phasespace_parser for details
            of the data columns.
        :param charge: float / None
            Charge of the beam.
        """
        pass

    @abstractmethod
    def _parse_phasespace(self, pfile):
        """Parse the phasespace file.

        :param str pfile: phasespace file name.

        :returns: data, charge.
        """
        raise NotImplementedError

    @abstractmethod
    def _parse_line(self):
        """Parse files which record beam evolutions..

        :returns: data.
        """
        raise NotImplementedError

    def reset(self):
        """Reset status and output files."""
        # input file
        with open(self._fin, 'w') as fp:
            fp.truncate()

        # output particle file
        with open(self._pout, 'w') as fp:
            fp.truncate()

        # Empty Line files and set LineParameter instances None
        for suffix in self._output_suffixes:
            with open(self._rootname + suffix, 'w') as fp:
                fp.truncate()

        self._out = None

        self._start = None
        self._end = None
        self._min = None
        self._max = None
        self._avg = None
        self._std = None

    @abstractmethod
    def _get_executable(self, parallel):
        raise NotImplementedError

    def _check_run(self, parallel=False):
        executable = find_executable(self._get_executable(parallel))
        if executable is None:
            raise RuntimeError(
               f"{executable} is not a valid bash command!")

        if not osp.isfile(self._fin):
            raise RuntimeError(f"Input file {self._fin} does not exist!")
        if not osp.getsize(self._fin):
            raise RuntimeError(f"Input file {self._fin} is empty!")

        return executable

    def _check_output(self):
        """Check output files."""
        pout = self._pout
        if not osp.isfile(pout):
            raise RuntimeError(f"Output particle file {pout} does not exist!")
        if not osp.getsize(pout):
            raise RuntimeError(f"Output particle file {pout} is empty!")

    def run(self, mapping, n_workers, timeout):
        """Run simulation for the beamline."""
        generate_input(self.template, mapping, self._fin)

        if not isinstance(n_workers, int) or not n_workers > 0:
            raise ValueError("n_workers must be a positive integer!")

        executable = self._check_run(n_workers > 1)
        command = f"{executable} {self._fin}"

        if n_workers > 1:
            command = f"mpirun -np {n_workers} " + command

        if timeout is not None:
            command = f"timeout {timeout}s " + command

        try:
            subprocess.check_output(command,
                                    stderr=subprocess.STDOUT,
                                    universal_newlines=True,
                                    shell=True,
                                    cwd=self._swd)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(repr(e))

        self._check_output()

        # TODO: process the data

    async def async_run(self, mapping, timeout):
        """Run simulation asynchronously for the beamline."""
        generate_input(self.template, mapping, self._fin)

        executable = self._check_run()
        command = f"{executable} {self._fin}"

        if timeout is not None:
            command = f"timeout {timeout}s " + command

        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._swd
        )

        _, stderr = await proc.communicate()
        if stderr:
            raise RuntimeError(stderr)

        self._check_output()

        # TODO: process the data

    def status(self):
        """Return the status of the beamline."""
        return {
            'out': self._out,
            'start': self._start,
            'end': self._end,
            'min': self._min,
            'max': self._max,
            'avg': self._avg,
            'std': self._std,
        }

    def __str__(self):
        text = 'Beamline: %s\n' % self.name
        text += f'Simulation working directory: {self._swd}\n'
        text += f'Input file: {osp.basename(self._fin)}\n'
        if self._pin is not None:
            text += 'Input particle file: %s\n' % self._pin
        if self._pout is not None:
            text += f'Output particle file: {osp.basename(self._pout)}\n'
        return text


class AstraBeamline(Beamline):
    """Beamline simulated using ASTRA."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._rootname = osp.join(
            self._swd, osp.basename(self._pout.split('.')[0]))

        self._output_suffixes = [
            '.Xemit.001', '.Yemit.001', '.Zemit.001', '.TRemit.001'
        ]

    def _get_executable(self, parallel):
        """Override."""
        if parallel:
            return config['EXECUTABLE_PARA']['ASTRA']
        return config['EXECUTABLE']['ASTRA']

    def _parse_phasespace(self, pfile):
        """Override."""
        return parse_astra_phasespace(pfile)

    def _parse_line(self):
        """Override."""
        return parse_astra_line(self._rootname)

    def generate_initial_particle_file(self, data, charge):
        """Implement the abstract method."""
        if self._pin is not None:
            ParticleFileGenerator(data, self._pin).to_astra_pfile(charge)


class ImpacttBeamline(Beamline):
    """Beamline simulated using IMPACT-T."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._pin = 'partcl.data'

        self._rootname = osp.join(self._swd, 'fort')

        if self._charge is None:
            raise ValueError(
                "Bunch charge is required for ImpactT simulation!")

        self._output_suffixes = ['.18', '.24', '.25', '.26']

    def _get_executable(self, parallel):
        """Override."""
        if parallel:
            return config['EXECUTABLE_PARA']['IMPACTT']
        return config['EXECUTABLE']['IMPACTT']

    def _parse_phasespace(self, pfile):
        """Override."""
        return parse_impactt_phasespace(pfile)

    def _parse_line(self):
        """Override."""
        return parse_impactt_line(self._rootname)

    def generate_initial_particle_file(self, data, charge):
        """Implement the abstract method."""
        if self._pin is not None:
            ParticleFileGenerator(data, self._pin).to_impactt_pfile()


def create_beamline(bl_type, *args, **kwargs):
    """Create and return a Beamline instance.

    :param str bl_type: beamline type
    """
    if bl_type.lower() in ('astra', 'a'):
        return AstraBeamline(*args, **kwargs)

    if bl_type.lower() in ('impactt', 't'):
        return ImpacttBeamline(*args, **kwargs)

    raise ValueError(f"Unknown beamline type {bl_type}!")
