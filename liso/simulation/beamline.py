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

from .input import (
    AstraInputGenerator, ImpacttInputGenerator, ElegantInputGenerator,
)
from ..config import config
from ..exceptions import LisoRuntimeError
from ..proc import (
    analyze_line,
    parse_astra_phasespace, parse_astra_line,
    parse_impactt_phasespace, parse_impactt_line,
    parse_elegant_phasespace, parse_elegant_line,
)
from ..simulation import ParticleFileGenerator
from ..io import TempSimulationDirectory


class Beamline(ABC):
    """Beamline abstraction class."""

    def __init__(self, name, *,
                 template=None,
                 swd=None,
                 fin=None,
                 pin=None,
                 pout=None,
                 charge=None,
                 z0=None):
        """Initialization.

        :param str name: Name of the beamline.
        :param str template: path of the template of the input file.
        :param str swd: path of the simulation working directory. This where
            the Python subprocess runs.
        :param str fin: input file name.
        :param str/None pin: initial particle file name.
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

        self._input_gen = self._parse_template_in(template)

        self._swd = osp.abspath('./' if swd is None else swd)

        self._fin = fin
        self._pin = pin
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
    def _parse_template_in(self, filepath):
        """Parse the template input file."""
        raise NotImplementedError

    def compile(self, mapping):
        self._input_gen.update(mapping)

    @abstractmethod
    def _generate_initial_particle_file(self, data, swd):
        """Generate the initial particle file.

        :param Phasespace data: particle phasespace.
        :param str swd: simulation working directory.
        """
        raise NotImplementedError

    @abstractmethod
    def _parse_phasespace(self, pfile):
        """Parse the phasespace file.

        :param str pfile: phasespace file name.

        :returns Phasespace: particle phasespace.
        """
        raise NotImplementedError

    @abstractmethod
    def _parse_line(self, rootname):
        """Parse files which record beam evolutions..

        :param str rootname: rootname of files.

        :returns: data.
        """
        raise NotImplementedError

    def reset(self):
        """Reset status and output files."""
        swd = self._swd

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
            raise LisoRuntimeError(f"{title} file {filepath} does not exist!")
        if not osp.getsize(filepath):
            raise LisoRuntimeError(f"{title} file {filepath} is empty!")

    def _check_executable(self, parallel=False):
        filepath = self._get_executable(parallel)
        executable = find_executable(filepath)
        assert executable is not None, \
            f"executable [{filepath}] is not available"
        return executable

    def _update_output(self, swd):
        """Analyse output particle file.

        Also prepare the initial particle file for the downstream simulation.

        :param str swd: simulation working directory.
        """
        pout = osp.join(swd, self._pout)
        self._check_file(pout, 'Output')

        ps = self._parse_phasespace(pout)
        if ps.charge is None:
            ps.charge = self._charge
        self._out = ps.analyze()
        return ps

    def _update_statistics(self):
        """Analysis output beam evolution files."""
        rootname = osp.join(self._swd, self._rootname)
        for suffix in self._output_suffixes:
            self._check_file(rootname + suffix, 'Output')

        data = self._parse_line(rootname)
        self._start = analyze_line(data, lambda x: x.iloc[0])
        self._end = analyze_line(data, lambda x: x.iloc[-1])
        self._min = analyze_line(data, np.min)
        self._max = analyze_line(data, np.max)
        self._avg = analyze_line(data, np.average)
        self._std = analyze_line(data, np.std)

    def _run_core(self, n_workers, timeout):
        executable = self._check_executable(n_workers > 1)

        # self._fin must be in the swd
        command = f"{executable} {self._fin}"
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
            raise LisoRuntimeError(repr(e))

    def run(self, phasespace, *, timeout, n_workers):
        """Run simulation for the beamline."""
        if not isinstance(n_workers, int) or not n_workers > 0:
            raise ValueError("n_workers must be a positive integer!")

        self.reset()

        if phasespace is not None:
            self._generate_initial_particle_file(phasespace, self._swd)

        # need absolute path here
        self._input_gen.write(osp.join(self._swd, self._fin))

        self._run_core(n_workers, timeout)

        self._update_statistics()

        return self._update_output(self._swd)

    async def _async_run_core(self, swd, timeout):
        executable = self._check_executable()

        # self._fin must be in the swd
        command = f"{executable} {self._fin}"
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
            # It does not raise even if command
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=out_file,
                stderr=asyncio.subprocess.PIPE,
                cwd=swd
            )

            _, err = await proc.communicate()

    async def async_run(self, phasespace, tmp_dir, *, timeout):
        """Run simulation asynchronously for the beamline."""
        with TempSimulationDirectory(osp.join(self._swd, tmp_dir),
                                     delete_old=True) as swd:

            if phasespace is not None:
                self._generate_initial_particle_file(phasespace, swd)

            # need absolute path here
            self._input_gen.write(osp.join(swd, self._fin))

            await self._async_run_core(swd, timeout)

            return self._update_output(swd)

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
        text += f'Input particle file: {self._pin}\n'
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

    def _parse_template_in(self, filepath):
        """Override."""
        return AstraInputGenerator(filepath)

    def _parse_phasespace(self, pfile):
        """Override."""
        return parse_astra_phasespace(pfile)

    def _parse_line(self, rootname):
        """Override."""
        return parse_astra_line(rootname)

    def _generate_initial_particle_file(self, data, swd):
        """Override."""
        ParticleFileGenerator.from_phasespace(data).to_astra(
            osp.join(swd, self._pin))


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

    def _parse_template_in(self, filepath):
        """Override."""
        return ImpacttInputGenerator(filepath)

    def _parse_phasespace(self, pfile):
        """Override."""
        return parse_impactt_phasespace(pfile)

    def _parse_line(self, rootname):
        """Override."""
        return parse_impactt_line(rootname)

    def _generate_initial_particle_file(self, data, swd):
        """Override."""
        ParticleFileGenerator.from_phasespace(data).to_impactt(
            osp.join(swd, self._pin))


class ElegantBeamline(Beamline):
    """Beamline simulated using ELEGANT."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._rootname = 'fort'

    def _get_executable(self, parallel):
        """Override."""
        if parallel:
            raise NotImplementedError
        return config['EXECUTABLE']['ELEGANT']

    def _parse_template_in(self, filepath):
        """Override."""
        return ElegantInputGenerator(filepath)

    def _parse_phasespace(self, pfile):
        """Override."""
        return parse_elegant_phasespace(pfile)

    def _parse_line(self, rootname):
        """Override."""
        raise NotImplementedError

    def _generate_initial_particle_file(self, data, swd):
        """Override."""
        ParticleFileGenerator.from_phasespace(data).to_elegant(
            osp.join(swd, self._pin))


def create_beamline(bl_type, *args, **kwargs):
    """Create and return a Beamline instance.

    :param str bl_type: beamline type
    """
    if bl_type.lower() in ('astra', 'a'):
        return AstraBeamline(*args, **kwargs)

    if bl_type.lower() in ('impactt', 't'):
        return ImpacttBeamline(*args, **kwargs)

    if bl_type.lower() in ('elegant', 'e'):
        return ElegantBeamline(*args, **kwargs)

    raise ValueError(f"Unknown beamline type {bl_type}!")
