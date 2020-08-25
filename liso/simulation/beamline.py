"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import asyncio
import os
import os.path as osp
from abc import ABC, abstractmethod
import subprocess
from distutils.spawn import find_executable

import numpy as np

from ..config import config
from ..data_processing import (
    analyze_beam, analyze_line,
    parse_astra_phasespace, parse_impactt_phasespace,
    parse_astra_line, parse_impactt_line,
)
from ..simulation import ParticleFileGenerator
from ..io import TempSimulationDirectory
from .output import OutputData
from .input import generate_input


class Beamline(ABC):
    """Beamline abstraction class."""

    def __init__(self, name, *,
                 template=None,
                 swd=None,
                 fin=None,
                 pout=None,
                 charge=None,
                 z0=None):
        """Initialization.

        :param str name: Name of the beamline.
        :param str template: path of the template of the input file.
        :param str swd: path of the simulation working directory. This where
            ASTRA expects to find all the data files (i.e. initial particle
            file and field files) specified in the input file. It should be
            noted that the input file does not have to be put in this
            directory.
        :param str fin: input file name.
        :param str pout: final particle file name. It must be located in the
            same directory as the input file.
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
            self._template = tuple(fp.readlines())

        self._swd = osp.abspath('./' if swd is None else swd)

        self._fin = fin
        self._pin = None  # Initial particle distribution file name.
        self._pout = pout
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

        self.next = None  # downstream beamline

    @property
    def out(self):
        return self._out

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    @property
    def avg(self):
        return self._avg

    @property
    def std(self):
        return self._std

    @abstractmethod
    def generate_initial_particle_file(self, data, charge):
        """Generate the initial particle file.

        :param data: Pandas.DataFrame
            Particle data. See data_processing/phasespace_parser for details
            of the data columns.
        :param charge: float / None
            Charge of the beam.
        """
        raise NotImplementedError

    @abstractmethod
    def _parse_phasespace(self, pfile):
        """Parse the phasespace file.

        :param str pfile: phasespace file name.

        :returns: data, charge.
        """
        raise NotImplementedError

    @abstractmethod
    def _parse_line(self, rootname):
        """Parse files which record beam evolutions..

        :param str rootname: rootname of files.

        :returns: data.
        """
        raise NotImplementedError

    def reset(self, tmp_dir=None):
        """Reset status and output files."""
        swd = self._swd if tmp_dir is None else \
            osp.join(os.getcwd(), tmp_dir)

        # input file
        with open(osp.join(swd, self._fin), 'w') as fp:
            fp.truncate()

        # output particle file
        with open(osp.join(swd, self._pout), 'w') as fp:
            fp.truncate()

        # Empty Line files and set LineParameter instances None
        for suffix in self._output_suffixes:
            with open(osp.join(swd, self._rootname + suffix), 'w') as fp:
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

    def _check_file(self, filepath, title=''):
        if not osp.isfile(filepath):
            raise RuntimeError(f"{title} file {filepath} does not exist!")
        if not osp.getsize(filepath):
            raise RuntimeError(f"{title} file {filepath} is empty!")

    def _check_run(self, parallel=False):
        executable = find_executable(self._get_executable(parallel))
        if executable is None:
            raise RuntimeError(
               f"{executable} is not a valid bash command!")
        return executable

    def _update_output(self, swd=None):
        """Analyse output particle file.

        Also prepare the input particle file for the downstream simulation.

        :param str swd: simulation working directory.
        """
        swd = self._swd if swd is None else swd
        pout = osp.join(swd, self._pout)
        self._check_file(pout, 'Output')

        data, charge = self._parse_phasespace(pout)
        charge = self._charge if charge is None else charge
        self._out = analyze_beam(data, charge)
        if self.next is not None:
            self.next.generate_initial_particle_file(data, charge)

        return data

    def _update_statistics(self, swd=None):
        """Analysis output beam evolution files.

        :param str swd: simulation working directory.
        """
        swd = self._swd if swd is None else swd
        rootname = osp.join(swd, self._rootname)
        for suffix in self._output_suffixes:
            self._check_file(rootname + suffix, 'Output')

        data = self._parse_line(rootname)
        self._start = analyze_line(data, lambda x: x.iloc[0])
        self._end = analyze_line(data, lambda x: x.iloc[-1])
        self._min = analyze_line(data, np.min)
        self._max = analyze_line(data, np.max)
        self._avg = analyze_line(data, np.average)
        self._std = analyze_line(data, np.std)

    def run(self, mapping, n_workers, timeout):
        """Run simulation for the beamline."""
        self.reset()

        fin = osp.join(self._swd, self._fin)
        generate_input(self._template, mapping, fin)

        if not isinstance(n_workers, int) or not n_workers > 0:
            raise ValueError("n_workers must be a positive integer!")

        self._check_file(fin, 'Input')
        executable = self._check_run(n_workers > 1)
        command = f"{executable} {fin}"
        if n_workers > 1:
            command = f"mpirun -np {n_workers} " + command

        if timeout is not None:
            command = f"timeout {timeout}s " + command

        try:
            # We do not want to generate a full history of the simulation
            # log. The current one is good enough for debugging.
            with open('simulation.log', "w") as out_file:
                subprocess.run(command,
                               stdout=out_file,
                               universal_newlines=True,
                               shell=True,
                               cwd=self._swd)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(repr(e))

        self._update_output()
        self._update_statistics()

    async def async_run(self, mapping, tmp_dir, *, timeout=None):
        """Run simulation asynchronously for the beamline."""
        with TempSimulationDirectory(osp.join(os.getcwd(), tmp_dir)) as swd:
            self.reset(tmp_dir)

            fin = osp.join(swd, self._fin)
            generate_input(self._template, mapping, fin)
            self._check_file(fin, 'Input')
            executable = self._check_run()

            command = f"{executable} {fin}"
            if timeout is not None:
                command = f"timeout {timeout}s " + command

            # Astra will find external files in the simulation working
            # directory but output files in the directory where the input
            # file is located.

            # We do not want to generate a full history of the simulation
            # log. The current one is good enough for debugging. It is not
            # a problem even if different processes write the file
            # interleavingly.
            with open(f'simulation.log', "w") as out_file:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=out_file,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self._swd
                )

            _, stderr = await proc.communicate()
            if stderr:
                raise RuntimeError(stderr)

            ps = self._update_output(swd)

            inputs = {
                f'{self.name}.{k}': v for k, v in mapping.items()
            }
            phasespaces = {
                f'{self.name}.out': ps
            }
            return OutputData(inputs, phasespaces)

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
            text += f'Input particle file: {self._pin}\n'
        if self._pout is not None:
            text += f'Output particle file: {self._pout}\n'
        return text


class AstraBeamline(Beamline):
    """Beamline simulated using ASTRA."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._rootname = osp.basename(self._pout.split('.')[0])

        self._output_suffixes = [
            '.Xemit.001', '.Yemit.001', '.Zemit.001'
        ]

    def _get_executable(self, parallel):
        """Override."""
        if parallel:
            return config['EXECUTABLE_PARA']['ASTRA']
        return config['EXECUTABLE']['ASTRA']

    def _parse_phasespace(self, pfile):
        """Override."""
        return parse_astra_phasespace(pfile)

    def _parse_line(self, rootname):
        """Override."""
        return parse_astra_line(rootname)

    def generate_initial_particle_file(self, data, charge):
        """Implement the abstract method."""
        pin = osp.join(self._swd, self._pin)
        if pin is not None:
            ParticleFileGenerator(data, pin).to_astra_pfile(charge)


class ImpacttBeamline(Beamline):
    """Beamline simulated using IMPACT-T."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._pin = 'partcl.data'
        self._rootname = 'fort'

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

    def _parse_line(self, rootname):
        """Override."""
        return parse_impactt_line(rootname)

    def generate_initial_particle_file(self, data, charge):
        """Implement the abstract method."""
        pin = osp.join(self._swd, self._pin)
        if pin is not None:
            ParticleFileGenerator.fromDataframeToImpactt(data, pin)


def create_beamline(bl_type, *args, **kwargs):
    """Create and return a Beamline instance.

    :param str bl_type: beamline type
    """
    if bl_type.lower() in ('astra', 'a'):
        return AstraBeamline(*args, **kwargs)

    if bl_type.lower() in ('impactt', 't'):
        return ImpacttBeamline(*args, **kwargs)

    raise ValueError(f"Unknown beamline type {bl_type}!")
