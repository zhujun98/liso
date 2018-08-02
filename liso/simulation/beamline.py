#!/usr/bin/python
"""
Author: Jun Zhu
"""
import os
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
import subprocess

import numpy as np

from .watch import AstraWatch, ImpacttWatch
from .line import AstraLine, ImpacttLine
from .input import InputGenerator
from ..data_processing import analyze_beam, analyze_line, tailor_beam, \
                              ParticleFileGenerator
from .simulation_utils import generate_input

from ..exceptions import *
from ..config import Config


INF = Config.INF


class Beamline(ABC):
    """Beamline abstraction class."""
    code = None  # code name
    exec_s = None  # series simulation exec
    exec_p = None  # parallel simulation exec

    def __init__(self, name, *,
                 gin=None,
                 fin=None,
                 template=None,
                 pin=None,
                 pout=None,
                 charge=None,
                 z0=None,
                 timeout=600):
        """Initialization.

        :param str name: Name of the beamline.
        :param InputGenerator gin: An input generator. If given, all
            the other keyword arguments are omitted.
        :param str fin: The path of the input file.
        :param str template: The path of the template file.
        :param str pin: Filename of the initial particle file.
        :param str pout: Filename of the output particle file.
        :param float charge: Bunch charge at the beginning of the beamline.
            Only used for certain codes (e.g. ImpactT).
        :param float z0: Starting z coordinate in meter. Used for concatenated
            simulation. Default = None, inherit z coordinate from the upstream
            beamline. However, for instance, when the second beamline is
            defined from z0 = 0.0, z0 is required to generate a correct initial
            particle distribution.
        :param float timeout: Maximum allowed duration in seconds of the
            simulation.
        """
        self.name = name
        if isinstance(gin, InputGenerator):
            self._gin = gin

        self.dirname = os.path.dirname(os.path.abspath(fin))
        self._fin = os.path.join(self.dirname, os.path.basename(fin))
        # Read the template file only once. Make the 'template' read-only by
        # converting it to a tuple.
        with open(template) as fp:
            self.template = tuple(fp.readlines())
        self.charge = charge

        self.next = None  # downstream beamline

        if pin is None:
            self.pin = pin
        else:
            self.pin = os.path.join(self.dirname, os.path.basename(pin))

        if pout is None:
            raise ValueError("Please specify the output particle file!")
        else:
            self.pout = os.path.join(self.dirname, os.path.basename(pout))

        self.z0 = z0  # starting z coordinate (m)

        # bookkeeping of Watch and the corresponding BeamParameters
        # {key: name of watch, value: (Watch, BeamParameters)}
        self._watches = OrderedDict()
        self.add_watch('out', self.pout)  # default Watch instance

        # suffixes for the output files related to Line instance.
        self._output_suffixes = []
        self._all = None  # Line  # default Line instance
        # default LineParameters instances
        self.max = None  # LineParameters
        self.min = None  # LineParameters
        self.ave = None  # LineParameters
        self.start = None  # LineParameters
        self.end = None  # LineParameters
        self.std = None  # LineParameters

        self._timeout = timeout

    def __getattr__(self, item):
        return self._watches[item][1]

    @abstractmethod
    def add_watch(self, name, pfile, **kwargs):
        """Add a Watch object to the beamline.

        :param name: string
            Name of the Watch object.
        :param pfile: string
            Filename of the Watch object. This file is assumed to be in
            the same folder of the input file of this object.
        """
        pass

    def generate_input(self, mapping):
        """Generate input file.

        :param mapping: dict
            A pattern-value mapping for replacing the pattern with
            value in the template file.
        """
        return generate_input(self.template, mapping, self._fin)

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

    def _check_watch(self, name, filename):
        """Check existence of a Watch instance.

        :param filename: string
            Path of the particle file for the new Watch instance.
        """
        if name in self._watches.keys():
            raise ValueError("Watch instance {} already exists!".format(name))
        return os.path.join(self.dirname, os.path.basename(filename))

    def clean(self):
        """Clean the output from the previous simulation.

        - Remove input files;
        - Remove files related to Watch and Line instances;
        - Set BeamParameters and LineParameters to None;
        - Remove initial particle files for a concatenate simulation.
        """
        with open(self._fin, 'w') as fp:
            fp.truncate()

        # Empty Watch particle files and set the corresponding
        # BeamParameter instances to None
        for item in self._watches.values():
            with open(item[0].pfile, 'w') as fp:
                fp.truncate()
            item[1] = None

        # Empty Line files and set LineParameter instances None
        for suffix in self._output_suffixes:
            with open(self._all.rootname + suffix, 'w') as fp:
                fp.truncate()
        self.max = None
        self.min = None
        self.start = None
        self.end = None
        self.ave = None
        self.std = None

        # Empty the initial particle file of the downstream beamline
        if self.next is not None:
            with open(self.next.pin, 'w') as fp:
                fp.truncate()

    def simulate(self, workers=1):
        """Simulate the beamline.

        :param int workers: Number of processes for parallel accelerator codes.
        """
        if workers > 1:
            command = "timeout {}s mpirun -np {} {} {} >/dev/null".format(
                self._timeout,
                workers,
                self.__class__.exec_p,
                self._fin)
        else:
            command = "timeout {}s {} {}".format(
                self._timeout,
                self.__class__.exec_s,
                os.path.basename(self._fin))

        # TODO: understand how to understand the simulation process.
        try:
            subprocess.check_output(command,
                                    stderr=subprocess.STDOUT,
                                    universal_newlines=True,
                                    shell=True,
                                    cwd=self.dirname)
        except subprocess.CalledProcessError as e:
            raise SimulationNotFinishedProperlyError(e)
        finally:
            time.sleep(1)

    def update_out(self):
        """Re-calculate BeamParameters at the final Watch - 'out'."""
        try:
            data, charge, self._watches['out'][1] = \
                self._process_watch(self._watches['out'][0])
        except Exception as e:
            raise LISOWatchUpdateError(e)

        if self.next is not None:
            self.next.generate_initial_particle_file(data, charge)

    def update_watches_and_lines(self):
        """Re-calculate all BeamParameters and LineParameters."""
        # Update watches
        # # 'out' will be processed in the method 'simulate()'
        for name, item in self._watches.items():
            if name != 'out':
                try:
                    _, _, item[1] = self._process_watch(item[0])
                except (FileNotFoundError, LISOFileEmptyError) as e:
                    raise LISOWatchUpdateError(e)

        # update lines
        try:
            data = self._all.load_data()

            self.start = analyze_line(data, lambda x: x.iloc[0])
            self.end = analyze_line(data, lambda x: x.iloc[-1])
            self.min = analyze_line(data, np.min)
            self.max = analyze_line(data, np.max)
            self.ave = analyze_line(data, np.average)
            self.std = analyze_line(data, np.std)
        except (FileNotFoundError, LISOFileEmptyError) as e:
            raise LISOLineUpdateError(e)

    def _process_watch(self, watch):
        """Process a Watch.

        Load the data and analyze the beam.

        :param watch: Watch.
            Watch instance.
        """
        data, charge = watch.load_data()

        # Even if charge is given for a AstraBeamline, it will still use
        # the charge returned from watch.load_data().
        charge = self.charge if charge is None else charge

        n0 = len(data)
        data = tailor_beam(data,
                           tail=watch.tail,
                           halo=watch.halo,
                           rotation=watch.rotation)

        charge *= len(data) / n0

        params = analyze_beam(data,
                              charge,
                              current_bins=watch.current_bins,
                              filter_size=watch.filter_size,
                              slice_percent=watch.slice_percent,
                              slice_with_peak_current=watch.slice_with_peak_current)

        return data, charge, params

    def __str__(self):
        text = 'Name: %s\n' % self.name
        text += 'Directory: %s\n' % self.dirname
        text += 'Input file: %s\n' % self._fin
        if self.pin is not None:
            text += 'Input particle file: %s\n' % self.pin
        if self.pout is not None:
            text += 'Output particle file: %s\n' % self.pout

        text += 'Watch point(s):\n'
        for item in self._watches.values():
            text += ' - ' + item[0].__str__()

        return text


def create_beamline(code, *args, **kwargs):
    """Create a Beamline instance.

    :param code: string
        Code name.
    """
    class AstraBeamline(Beamline):
        """Beamline simulated by ASTRA.

        Inherit from Beamline class.
        """
        code = 'a'
        exec_s = Config.ASTRA
        exec_p = Config.ASTRA_P

        def __init__(self, *args, **kwargs):
            """Initialization."""
            super().__init__(*args, **kwargs)

            rootpath = os.path.join(self.dirname,
                                    os.path.basename(self.pout.split('.')[0]))
            self._all = AstraLine('all', rootpath)
            self._output_suffixes = ['.Xemit.001', '.Yemit.001', '.Zemit.001',
                                     '.TRemit.001']

        def add_watch(self, name, pfile, **kwargs):
            """Implement the abstract method."""
            pfile = self._check_watch(name, pfile)
            self._watches[name] = [AstraWatch(name, pfile, **kwargs), None]

        def generate_initial_particle_file(self, data, charge):
            """Implement the abstract method."""
            if self.pin is not None:
                ParticleFileGenerator(data, self.pin).to_astra_pfile(charge)

    class ImpacttBeamline(Beamline):
        """Beamline simulated by IMPACT-T.

        Inherit from Beamline class.
        """
        code = 't'
        exec_s = Config.IMPACTT
        exec_p = Config.IMPACTT_P

        def __init__(self, pin='partcl.data', *args, **kwargs):
            """Initialization."""
            super().__init__(pin=pin, *args, **kwargs)
            if self.pin is not None and os.path.basename(
                    self.pin) != 'partcl.data':
                raise ValueError(
                    "Input particle file for ImpactT must be 'partcl.data'!")

            if self.charge is None:
                raise ValueError(
                    "Bunch charge is required for ImpactT simulation!")

            rootpath = os.path.join(self.dirname,
                                    os.path.basename(self.pout.split('.')[0]))
            self._all = ImpacttLine('all', rootpath)
            self._output_suffixes = ['.18', '.24', '.25', '.26']

        def add_watch(self, name, pfile, **kwargs):
            """Implement the abstract method."""
            pfile = self._check_watch(name, pfile)
            self._watches[name] = [ImpacttWatch(name, pfile, **kwargs), None]

        def generate_initial_particle_file(self, data, charge):
            """Implement the abstract method."""
            if self.pin is not None:
                ParticleFileGenerator(data, self.pin).to_impactt_pfile()

    class ImpactzBeamline(Beamline):
        """Beamline simulated by IMPACT-Z.

        Inherit from Beamline class.
        """
        code = 'z'
        exec_s = None
        exec_p = None
        pass

    class GenesisBeamline(Beamline):
        """Beamline simulated by GENESIS.

        Inherit from Beamline class.
        """
        code = 'g'
        exec_s = None
        exec_p = None
        pass

    if code.lower() in ('astra', 'a'):
        return AstraBeamline(*args, **kwargs)

    if code.lower() in ('impactt', 't'):
        return ImpacttBeamline(*args, **kwargs)

    if code.lower() in ('impactz', 'z'):
        return ImpactzBeamline(*args, **kwargs)

    if code.lower() in ('genesis', 'g'):
        return GenesisBeamline(*args, **kwargs)

    raise ValueError("Unknown code!")
