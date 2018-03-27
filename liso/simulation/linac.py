#!/usr/bin/python
"""
Author: Jun Zhu
"""
import os
from collections import defaultdict

from .smlt_utils import generate_input
from .beamline import create_beamline
from ..exceptions import *


class Linac(object):
    """Linac class.

    The linac class is the abstraction of a linear accelerator. It
    consists of one or multiple beamlines. These beamlines can be
    simulated using different codes.
    """
    def __init__(self):
        """Initialization."""
        self._beamlines = defaultdict()

    def __getattr__(self, item):
        return self._beamlines[item]

    def add_beamline(self, code, *args, **kwargs):
        """Add a beamline.

        :param code: string.
            Code used to simulate the beamline.
        """
        bl = create_beamline(code, *args, **kwargs)

        if bl.name in self._beamlines.keys():
            raise ValueError("Beamline named {} already exists!".format(bl.name))

        # connect the new Beamline to the last added Beamline
        if len(self._beamlines) != 0:
            bl.predecessor = list(self._beamlines.values())[-1]

            if bl.pin is None or bl.predecessor.pout is None:
                raise ValueError("Input particle of the new Beamline and output "
                                 "particle file of the predecessor Beamline must "
                                 "be specified for a concatenated simulation!")

        self._beamlines[bl.name] = bl

    def add_watch(self, beamline, *args, **kwargs):
        """Add a Watch object to a Beamline of the Linac.

        :param beamline: string
            Name of the Beamline object.
        """
        self._beamlines[beamline].add_watch(*args, **kwargs)

    def update(self, x_map, workers=1):
        """Update the linac.

        Re-simulate all beamlines and update all BeamParameters and
        LineParameters.

        :param: x_map: dict
            A dictionary for variables and covariables- {name: value}.
        :param workers: int
            Number of threads.

        :return: A bool value indicates whether the output files have
                 been produced correctly.
        """
        # First clean all the previous output
        for beamline in self._beamlines.values():
            beamline.clean()

        # Generate new input files for next simulation
        found = set()
        for bl in self._beamlines.values():
            try:
                os.remove(os.path.join(bl.dirname, bl.fin))
            except FileNotFoundError:
                pass
            found = found.union(generate_input(bl, x_map))

            fin = os.path.join(bl.dirname, bl.fin)
            if not os.path.exists(fin):
                raise BeamlineInputFileNotFoundError(
                    "The input file %s has not been generated!" % fin)

        if found != x_map.keys():
            raise ValueError("Variables %s not found in the templates!" %
                             (x_map.keys() - found))

        # Run simulations, and update all the BeamParameters and LineParameters
        for (i, beamline) in enumerate(self._beamlines.values()):
            # TODO: implement multi-threading here
            beamline.simulate(workers)
            beamline.update()

    def __str__(self):
        text = ''
        for beamline in self._beamlines.values():
            text += beamline.__str__() + '\n'

        return text
