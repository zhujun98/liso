"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import os
import os.path as osp
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
import subprocess
from distutils.spawn import find_executable

import numpy as np

from ..config import config
from .input import InputGenerator
from ..data_processing import (
    analyze_beam, analyze_line, tailor_beam, ParticleFileGenerator,
    parse_astra_phasespace, parse_impactt_phasespace,
    parse_astra_line, parse_impactt_line,
)
from .simulation_utils import generate_input

from ..exceptions import *


class Beamline(ABC):
    """Beamline abstraction class."""

    def __init__(self, name, *,
                 gin=None,
                 fin=None,
                 template=None,
                 pin=None,
                 pout=None,
                 charge=None,
                 z0=None):
        """Initialization.

        :param str name: Name of the beamline.
        :param InputGenerator gin: An input generator. If given, all
            the other keyword arguments are omitted.
        :param str fin: The path of the input file.
        :param str template: The path of the template file.
        :param str pin: name of the initial particle file.
        :param str pout: name of the output particle file.
        :param float charge: Bunch charge at the beginning of the beamline.
            Only used for certain codes (e.g. ImpactT).
        :param float z0: Starting z coordinate in meter. Used for concatenated
            simulation. Default = None, inherit z coordinate from the upstream
            beamline. However, for instance, when the second beamline is
            defined from z0 = 0.0, z0 is required to generate a correct initial
            particle distribution.
        """
        self.name = name
        if isinstance(gin, InputGenerator):
            self._gin = gin

        self._fin = fin

        # Read the template file only once. Make the 'template' read-only by
        # converting it to a tuple.
        with open(template) as fp:
            self.template = tuple(fp.readlines())
        self._charge = charge

        self._pin = pin
        self._pout = pout

        self.z0 = z0  # starting z coordinate (m)

        self._out = None  # BeamParameters

        # suffixes for the output files related to Line instance.
        self._output_suffixes = []

        self._max = None  # LineParameters
        self._min = None  # LineParameters
        self._ave = None  # LineParameters
        self._start = None  # LineParameters
        self._end = None  # LineParameters
        self._std = None  # LineParameters

    @property
    def out(self):
        if self._out is None:
            data, charge = self._parse_phasespace(self._pout)
            charge = self._charge if charge is None else charge
            self._out = analyze_beam(data, charge)
        return None

    @property
    def start(self):
        if self._start is None:
            self._update_statistics()
            return self._start
        return None

    @property
    def end(self):
        if self._end is None:
            self._update_statistics()
            return self._end
        return None

    @property
    def min(self):
        if self._min is None:
            self._update_statistics()
            return self._min
        return None

    @property
    def max(self):
        if self._max is None:
            self._update_statistics()
            return self._max
        return None

    @property
    def ave(self):
        if self._ave is None:
            self._update_statistics()
            return self._ave
        return None

    @property
    def std(self):
        if self._std is None:
            self._update_statistics()
            return self._std
        return None

    def _update_statistics(self):
        data = self._parse_line('...')

        self._start = analyze_line(data, lambda x: x.iloc[0])
        self._end = analyze_line(data, lambda x: x.iloc[-1])
        self._min = analyze_line(data, np.min)
        self._max = analyze_line(data, np.max)
        self._ave = analyze_line(data, np.average)
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
    def _parse_line(self, rootname):
        """Parse files which record beam evolutions..

        :param str rootname: rootname of those files.

        :returns: data.
        """
        raise NotImplementedError

    def reset(self):
        """Reset output parameters."""
        self._out = None

        self._max = None
        self._min = None
        self._start = None
        self._end = None
        self._ave = None
        self._std = None

    def run(self, mapping, n_workers, timeout, cwd):
        """Run simulation for the beamline."""
        generate_input(self.template, mapping, self._fin)

        if not isinstance(n_workers, int) or not n_workers > 0:
            raise ValueError("n_workers must be a positive integer!")

        if n_workers > 1:
            executable = find_executable(config['EXECUTABLE_PARA']['ASTRA'])
        else:
            executable = find_executable(config['EXECUTABLE']['ASTRA'])

        if executable is None:
            raise CommandNotFoundError(
               f"{executable} is not a valid bash command")
        else:
            command = f"{executable} {self._fin}"

        if n_workers > 1:
            command = f"mpirun -np {n_workers} " + command

        if timeout is not None:
            command = f"timeout {timeout}s " + command

        if not os.path.isfile(self._fin):
            raise InputFileNotFoundError(self._fin + " does not exist!")
        if not os.path.getsize(self._fin):
            raise InputFileEmptyError(self._fin + " is empty!")
        if cwd is None:
            cwd = os.path.dirname(os.path.abspath(self._fin))

        try:
            subprocess.check_output(command,
                                    stderr=subprocess.STDOUT,
                                    universal_newlines=True,
                                    shell=True,
                                    cwd=cwd)
        except subprocess.CalledProcessError as e:
            raise SimulationNotFinishedProperlyError(e)
        finally:
            time.sleep(1)

    def __str__(self):
        text = 'Name: %s\n' % self.name
        text += 'Input file: %s\n' % self._fin
        if self._pin is not None:
            text += 'Input particle file: %s\n' % self._pin
        if self._pout is not None:
            text += 'Output particle file: %s\n' % self._pout
        return text


def create_beamline(bl_type, *args, **kwargs):
    """Create and return a Beamline instance.

    :param str bl_type: beamline type
    """
    class AstraBeamline(Beamline):
        """Beamline simulated using ASTRA."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self._output_suffixes = ['.Xemit.001', '.Yemit.001', '.Zemit.001',
                                     '.TRemit.001']

        def _parse_phasespace(self, pfile):
            """Override."""
            return parse_astra_phasespace(pfile)

        def _parse_line(self, rootname):
            """Override."""
            return parse_astra_line(rootname)

        def generate_initial_particle_file(self, data, charge):
            """Implement the abstract method."""
            if self._pin is not None:
                ParticleFileGenerator(data, self._pin).to_astra_pfile(charge)

    class ImpacttBeamline(Beamline):
        """Beamline simulated using IMPACT-T."""
        def __init__(self, pin='partcl.data', *args, **kwargs):
            super().__init__(pin=pin, *args, **kwargs)

            if self._pin is not None and os.path.basename(
                    self._pin) != 'partcl.data':
                raise ValueError(
                    "Input particle file for ImpactT must be 'partcl.data'!")

            if self._charge is None:
                raise ValueError(
                    "Bunch charge is required for ImpactT simulation!")

            self._output_suffixes = ['.18', '.24', '.25', '.26']

        def _parse_phasespace(self, pfile):
            """Override."""
            return parse_impactt_phasespace(pfile)

        def _parse_line(self, rootname):
            """Override."""
            return parse_impactt_line(rootname)

        def generate_initial_particle_file(self, data, charge):
            """Implement the abstract method."""
            if self._pin is not None:
                ParticleFileGenerator(data, self._pin).to_impactt_pfile()

    if bl_type.lower() in ('astra', 'a'):
        return AstraBeamline(*args, **kwargs)

    if bl_type.lower() in ('impactt', 't'):
        return ImpacttBeamline(*args, **kwargs)

    raise ValueError(f"Unknown beamline type {bl_type}!")
