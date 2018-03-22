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

        :param val: string.
            Code used to simulate the beamline.
        """
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

        if bl.name in self.beamlines.keys():
            raise ValueError("Beamline named {} already exists!".format(bl.name))

        # connect the new Beamline to the last added Beamline
        if len(self.beamlines) != 0:
            bl.predecessor = list(self.beamlines.values())[-1]

            if bl.pin is None or bl.predecessor.pout is None:
                raise ValueError("Input particle of the new Beamline and output "
                                 "particle file of the predecessor Beamline must "
                                 "be specified for a concatenated simulation!")

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

    def update(self, var_dict, workers=1):
        """Update the linac.

        Re-simulate the beamlines and update all FitPoints and FitLines.

        :param: var_dict: dict
            A dictionary for variables with key=name and value=value.
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
            except FileNotFoundError:
                pass
            found = found.union(generate_input(bl, var_dict))

            fin = os.path.join(bl.dirname, bl.fin)
            if not os.path.exists(fin):
                raise BeamlineInputFileNotFoundError(
                    "The input file %s has not been generated!" % fin)

        if found != var_dict.keys():
            raise ValueError("Variables %s not found in the templates!" %
                             (var_dict.keys() - found))

        # Remove files generated in the last simulation.
        # Not raise.
        for beamline in self.beamlines.values():
            beamline.clean()

        # Run simulations.
        # Raise SimulationNotFinishedProperlyError
        for (i, beamline) in enumerate(self.beamlines.values()):
            if i != 0:
                beamline.update_pin()
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
