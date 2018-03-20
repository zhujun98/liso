#!/usr/bin/python
"""
Author: Jun Zhu
"""
import os
import time
from abc import ABC
from abc import abstractmethod
from collections import OrderedDict

import subprocess

from .watch import AstraWatch, ImpacttWatch
from .line import AstraLine, ImpacttLine
from .input import InputGenerator
from ..data_processing import analyze_beam, analyze_line
from ..data_processing import convert_particle_file

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
                 z0=0.0):
        """Initialization.

        :param name: string
            Name of the beamline.
        :param gin: InputGenerator object
            An input generator. If given, all the other keyword
            arguments are omitted.
        :param fin: string
            The path of the input file.
        :param template: string
            The path of the template file.
        :param pin: string
            Filename of the initial particle file.
        :param pout: string
            Filename of the output particle file.
        :param charge: float
            Bunch charge at the beginning of the beamline. Only used
            for certain codes (e.g. Impact-T).
        :param z0: float
            Starting z coordinate in meter. Used for concatenated
            simulation, e.g. Suppose the second beamline is defined
            from z0 = 0.0, when doing the particle file conversion,
            z0 is required.
        """
        self.name = name
        if isinstance(gin, InputGenerator):
            self._gin = gin

        self.dirname = os.path.dirname(os.path.abspath(fin))
        self.fin = os.path.basename(fin)
        # Read the template file only once.
        with open(template) as fp:
            self.template = tuple(fp.readlines())
        self.charge = charge

        self.predecessor = None  # the beamline upstream
        self.pin = pin
        self.pout = pout
        self.z0 = z0  # starting z coordinate (m)

        self.watches = OrderedDict()
        self.lines = OrderedDict()
        self._output_suffixes = []

    @abstractmethod
    def add_watch(self, name, pfile, **kwargs):
        """Add a Watch object to the beamline.

        :param name: string
            Name of the Watch object.
        :param pfile: string
            Filename of the Watch object. This file is assumed to be in
            the same folder of the input file of this object.
        """
        if hasattr(self, name):
            raise ValueError("Watch attribute {} already exists!".format(name))
        return os.path.join(self.dirname, os.path.basename(pfile))

    @abstractmethod
    def add_line(self, name, rootname, **kwargs):
        """Add a Line object to the beamline.

        :param name: string
            Name of the Line object.
        :param rootname
            Rootname of the Line object. Files with the rootname are assumed
            to be in the same folder of the input file of this object.
        """
        if hasattr(self, name):
            raise ValueError("Line attribute {} already exists!".format(name))
        return os.path.join(self.dirname, os.path.basename(rootname))

    def clean(self):
        """Delete the files bound to all FitPoints and FitLines."""
        for watch in self.watches.values():
            try:
                os.remove(watch.pfile)
            except OSError:
                pass

        for line in self.lines.values():
            for suffix in self._output_suffixes:
                try:
                    os.remove(line.rootname + suffix)
                except OSError:
                    pass

    def update(self):
        """Re-calculation for all the Watches and Lines.

        :return: A flag indicating whether the update fails. For example:
            corresponding file is not found.
        """
        for watch in self.watches.values():
            params = None
            try:
                data, charge = watch.load_data()
                # Even if charge is given for a AstraBeamline, it will still use
                # the charge returned from watch.load_data().
                charge = self.charge if charge is None else charge
                params = analyze_beam(data, charge,
                                      cut_halo=watch.cut_halo,
                                      cut_tail=watch.cut_tail,
                                      current_bins=watch.current_bins,
                                      filter_size=watch.filter_size,
                                      slice_percent=watch.slice_percent,
                                      slice_with_peak_current=watch.slice_with_peak_current)
            except FileNotFoundError:
                raise
            except Exception as e:
                print(e)
                raise
            finally:
                super().__setattr__(watch.name, params)

        for line in self.lines.values():
            params = None
            try:
                data = line.load_data()
                params = analyze_line(data, line.zlim)
            except FileNotFoundError:
                raise
            except Exception as e:
                print(e)
                raise
            finally:
                super().__setattr__(line.name, params)

    def update_pin(self):
        """Update the input particle file."""
        if self.predecessor is None:
            return

        pout = os.path.join(self.predecessor.dirname, os.path.basename(self.predecessor.pout))
        pin = os.path.join(self.dirname, os.path.basename(self.pin))

        os.remove(pin)
        convert_particle_file(pout, pin,
                              code_pout=self.predecessor.code,
                              code_pin=self.code,
                              z0=self.z0)

    def simulate(self, workers=1, time_out=300):
        """Simulate the beamline.

        :param workers: int
            Number of threads.
        :param time_out: int
            Number of seconds.
        """
        if not os.path.exists(os.path.join(self.dirname, self.fin)):
            pass

        if workers > 1:
            command = "timeout {}s mpirun -np {} {} {} >/dev/null".format(
                time_out,
                workers,
                self.__class__.exec_p,
                self.fin)
        else:
            command = "timeout {}s {} {}".format(
                time_out,
                self.__class__.exec_s,
                os.path.basename(self.fin))

        # TODO: understand how to understand the simulation process.
        try:
            output = subprocess.check_output(command,
                                             stderr=subprocess.STDOUT,
                                             universal_newlines=True,
                                             shell=True,
                                             cwd=self.dirname)
        except subprocess.CalledProcessError as e:
            print(e)
            raise SimulationNotFinishedProperlyError(e)
        except Exception as e:
            print(e)
        finally:
            time.sleep(1)

    def __str__(self):
        text = '\nBeamline: %s\n' % self.name
        text += '=' * 80 + '\n'
        text += 'Directory: %s\n' % self.dirname
        text += 'Input file: %s\n' % self.fin
        if self.pin is not None:
            text += 'Input particle file: %s\n' % self.pin
        if self.pout is not None:
            text += 'Output particle file: %s\n' % self.pout

        text += '\nWatch points:\n'
        if not self.watches:
            text += 'None'
        else:
            for watch in self.watches.values():
                text += watch.__str__()

        text += '\nLine sections:\n'
        if not self.lines:
            text += 'None'
        else:
            for line in self.lines.values():
                text += line.__str__()

        return text


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
        self._output_suffixes = ['.Xemit.001', '.Yemit.001', '.Zemit.001', '.TRemit.001']

    def add_watch(self, name, pfile, **kwargs):
        """Override the abstract method."""
        pfile = super().add_watch(name, pfile, **kwargs)
        self.watches[name] = AstraWatch(name, pfile, **kwargs)

    def add_line(self, name, rootname, **kwargs):
        """Override the abstract method."""
        rootname = super().add_line(name, rootname, **kwargs)
        self.lines[name] = AstraLine(name, rootname, **kwargs)


class ImpacttBeamline(Beamline):
    """Beamline simulated by IMPACT-T.

    Inherit from Beamline class.
    """
    code = 't'
    exec_s = Config.IMPACTT
    exec_p = Config.IMPACTT_P

    def __init__(self, *args, **kwargs):
        """Initialization."""
        super().__init__(*args, **kwargs)
        if self.pin is not None and self.pin != 'partcl.data':
            print("Warning: input particle file for Impact-T must be 'partcl.data'!")
        self.pin = 'partcl.data'

        if self.charge is None:
            raise ValueError("Bunch charge is required for Impact-T simulation!")
        self._output_suffixes = ['.18', '.24', '.25', '.26']

    def add_watch(self, name, pfile, **kwargs):
        """Override the abstract method."""
        pfile = super().add_watch(name, pfile, **kwargs)
        self.watches[name] = ImpacttWatch(name, pfile, **kwargs)

    def add_line(self, name, rootname, **kwargs):
        """Override the abstract method."""
        rootname = super().add_line(name, rootname, **kwargs)
        self.lines[name] = ImpacttLine(name, rootname, **kwargs)


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
