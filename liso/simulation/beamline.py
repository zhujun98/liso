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

from ..backend import config
from .watch import AstraWatch, ImpacttWatch
from .line import AstraLine, ImpacttLine

from ..exceptions import *

INF = config['INF']


class Beamline(ABC):
    """Beamline abstraction class."""
    exec_s = None  # series simulation exec
    exec_p = None  # parallel simulation exec

    def __init__(self, name, input_file, template):
        """Initialization.

        :param name: string
            Name of the beamline.
        :param input_file: string
            The path of the input file.
        :param template: string
            The path of the template file.
        """
        self.name = name
        self.dirname = os.path.dirname(os.path.abspath(input_file))
        self.input_file = os.path.basename(input_file)
        # Read the template file only once.
        with open(template) as fp:
            self.template = tuple(fp.readlines())

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
        :param kwargs: keyword arguments
            Pass to the initializer of the Watch object.
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
        :param kwargs: keyword arguments
            Pass to the initializer of the Line object.
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
            data = None
            try:
                data = watch.get_data()
            except FileNotFoundError:
                raise
            except Exception as e:
                print(e)
                raise
            finally:
                super().__setattr__(watch.name, data)

        for line in self.lines.values():
            data = None
            try:
                data = line.get_data()
            except FileNotFoundError:
                raise
            except Exception as e:
                print(e)
                raise
            finally:
                super().__setattr__(line.name, data)

    def simulate(self, workers=1, time_out=300):
        """Simulate the beamline.

        :param workers: int
            Number of threads.
        :param time_out: int
            Number of seconds.
        """
        if not os.path.exists(os.path.join(self.dirname, self.input_file)):
            pass

        if workers > 1:
            command = "timeout {}s mpirun -np {} {} {} >/dev/null".format(
                time_out,
                workers,
                self.__class__.exec_p,
                self.input_file)
        else:
            command = "timeout {}s {} {}".format(
                time_out,
                self.__class__.exec_s,
                os.path.basename(self.input_file))

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
        text += 'Input file: %s\n' % self.input_file

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

        text += '\n' + '=' * 80

        return text


class AstraBeamline(Beamline):
    """Beamline simulated by ASTRA.

    Inherit from Beamline class.
    """
    exec_s = config['ASTRA']
    exec_p = config['ASTRA_P']

    def __init__(self, name, input_file, template):
        """Initialization."""
        super().__init__(name, input_file, template)
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
    exec_s = config['IMPACTT']
    exec_p = config['IMPACTT_P']

    def __init__(self, name, input_file, template, charge):
        """Initialization."""
        super().__init__(name, input_file, template)
        self.charge = charge
        self._output_suffixes = ['.18', '.24', '.25', '.26']

    def add_watch(self, name, pfile, **kwargs):
        """Override the abstract method."""
        pfile = super().add_watch(name, pfile, **kwargs)
        self.watches[name] = ImpacttWatch(name, pfile, self.charge, **kwargs)

    def add_line(self, name, rootname, **kwargs):
        """Override the abstract method."""
        rootname = super().add_line(name, rootname, **kwargs)
        self.lines[name] = ImpacttLine(name, rootname, **kwargs)


class ImpactzBeamline(Beamline):
    """Beamline simulated by IMPACT-Z.

    Inherit from Beamline class.
    """
    exec_s = None
    exec_p = None
    pass


class GenesisBeamline(Beamline):
    """Beamline simulated by GENESIS.

    Inherit from Beamline class.
    """
    exec_s = None
    exec_p = None
    pass
