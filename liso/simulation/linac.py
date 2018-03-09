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

    def add_beamline(self, val, *args, **kwargs):
        """Add a beamline.

        :param val: string or Beamline object.
            Code used to simulate the beamline or an existing beamline.
        """
        if isinstance(val, Beamline):
            bl = val
        else:
            if val.lower() in ('astra', 'a'):
                bl = AstraBeamline(*args, **kwargs)
            elif val.lower() in ('impactt', 't'):
                bl = ImpacttBeamline(*args, **kwargs)
            elif val.lower() in ('impactz', 'z'):
                bl = ImpactzBeamline(*args, **kwargs)
            elif val.lower() in ('genesis', 'g'):
                bl = GenesisBeamline(*args, **kwargs)
            else:
                raise ValueError("Unknown code!")

        self.beamlines[bl.name] = bl

    def add_watch(self, beamline, *args, **kwargs):
        """Add a Watch object to a Beamline of the Linac.

        :param beamline: string
            Name of the Beamline object.
        """
        self.beamlines[beamline].add_watch(*args, **kwargs)

    def add_line(self, beamline, *args, **kwargs):
        """Add a Line object to a Beamline of the Linac.

        :param beamline: string
            Name of the Beamline object.
        """
        self.beamlines[beamline].add_line(*args, **kwargs)

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
                os.remove(os.path.join(bl.dirname, bl.fin))
            except OSError:
                pass
            found = found.union(generate_input(bl, mapping))

            fin = os.path.join(bl.dirname, bl.fin)
            if not os.path.exists(fin):
                raise BeamlineInputFileNotGeneratedError("Not found: %s" % fin)

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
