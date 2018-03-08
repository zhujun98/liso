#!/usr/bin/python
"""
Author: Jun Zhu
"""
from collections import defaultdict

from .smlt_utils import generate_input
from .beamline import *
from ..exceptions import *


class Linac(object):
    """Linac class.

    The linac class is the abstraction of a linear accelerator. It
    consists of one or multiple beamlines. These beamlines can be
    simulated using different codes.
    """
    def __init__(self):
        """Initialization."""
        self.beamlines = defaultdict()

    def add_beamline(self, *,
                     code=None,
                     name=None,
                     input_file=None,
                     template=None,
                     charge=0.0):
        """Add a beamline.

        :param code: string
            Code used to simulate the beamline.
        """
        if not isinstance(code, str):
            raise TypeError("code must be a string!")
        if not isinstance(name, str):
            raise TypeError("name must be a string!")
        if not isinstance(input_file, str):
            raise TypeError("input_file must be a string!")
        if not isinstance(template, str):
            raise TypeError("template must be a string!")
        if not isinstance(charge, (int, float)):
            raise TypeError("charge must be an integer or a float!")

        if code.lower() in ('astra', 'a'):
            beamline = AstraBeamline(name, input_file, template)
        elif code.lower() in ('impactt', 'i'):
            beamline = ImpacttBeamline(name, input_file, template, charge)
        elif code.lower() in ('impactz', 'z'):
            beamline = ImpactzBeamline(name, input_file, template)
        elif code.lower() in ('genesis', 'g'):
            beamline = GenesisBeamline(name, input_file, template)
        else:
            raise ValueError("Unknown code!")

        self.beamlines[beamline.name] = beamline

    def add_watch(self, *, beamline=None, name=None, pfile=None, **kwargs):
        """Add a Watch object to a Beamline of the Linac.

        :param beamline: string
            Name of the Beamline object.
        :param name: string
            Name of the Watch object.
        :param pfile: string

        :param kwargs: keyword arguments.
            Pass to add_watch() method of the Beamline class.
        """
        if not isinstance(beamline, str):
            raise TypeError("beamline must be a string!")
        if not isinstance(name, str):
            raise TypeError("name must be a string!")
        if not isinstance(pfile, str):
            raise TypeError("pfile must be a string!")

        self.beamlines[beamline].add_watch(name, pfile, **kwargs)

    def add_line(self, *, beamline=None, name=None, rootname=None, **kwargs):
        """Add a Line object to a Beamline of the Linac.

        :param beamline: string
            Name of the Beamline object.
        :param name: string
            Name of the Line object.
        :param rootname: string
            Name
        :param kwargs: keyword arguments
            Pass to add_line() method of the Beamline class.
        """
        if not isinstance(beamline, str):
            raise TypeError("beamline must be a string!")
        if not isinstance(name, str):
            raise TypeError("name must be a string!")
        if not isinstance(rootname, str):
            raise TypeError("rootname must be a string!")

        self.beamlines[beamline].add_line(name, rootname, **kwargs)

    def update(self, mapping, workers=1):
        """Update the linac.

        Re-simulate the beamlines and update all FitPoints and FitLines.

        :param: mapping

        :param workers: int
            Number of threads.

        :return: A bool value indicates whether the output files have
                 been produced correctly.
        """
        # Generate new input files for next simulation
        found = set()
        for bl in self.beamlines.values():
            try:
                os.remove(os.path.join(bl.dirname, bl.input_file))
            except OSError:
                pass
            found = found.union(generate_input(bl, mapping))

            input_file = os.path.join(bl.dirname, bl.input_file)
            if not os.path.exists(input_file):
                raise BeamlineInputFileNotGeneratedError("Not found: %s" % input_file)

        if found != mapping.keys():
            raise ValueError("Variables %s not found in the templates!" %
                             (mapping.keys() - found))

        # Remove files generated in the last simulation.
        # Not raise.
        for beamline in self.beamlines.values():
            beamline.clean()

        # Run simulations.
        # Raise SimulationNotFinishedProperlyError
        for beamline in self.beamlines.values():
            beamline.simulate(workers)

        # Recalculate the parameters at each FitPoint and FitLine.
        # Raise FileNotFoundError
        for beamline in self.beamlines.values():
            beamline.update()

    def __str__(self):
        text = ''
        for beamline in self.beamlines.values():
            text += beamline.__str__() + '\n'

        return text
