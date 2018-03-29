#!/usr/bin/python
"""
Author: Jun Zhu
"""
from collections import OrderedDict

from .beamline import create_beamline


class Linac(object):
    """Linac class.

    The linac class is the abstraction of a linear accelerator. It
    consists of one or multiple beamlines. These beamlines can be
    simulated using different codes.
    """
    def __init__(self):
        """Initialization."""
        self._beamlines = OrderedDict()

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
            list(self._beamlines.values())[-1].next = bl

            if bl.pin is None:
                raise ValueError("Initial particle file of the new Beamline must "
                                 "be specified for a concatenated simulation!")
        print(bl.name)
        self._beamlines[bl.name] = bl

    def add_watch(self, beamline, *args, **kwargs):
        """Add a Watch object to a Beamline of the Linac.

        :param beamline: string
            Name of the Beamline object.
        """
        self._beamlines[beamline].add_watch(*args, **kwargs)

    def update(self, mapping, workers=1):
        """Update the linac.

        Re-simulate all beamlines and update all BeamParameters and
        LineParameters.

        :param: mapping: dict
            A dictionary for variables and covariables- {name: value}.
        :param workers: int
            Number of threads.

        :return: A bool value indicates whether the output files have
                 been produced correctly.
        """
        # First clean all the previous output
        for beamline in self._beamlines.values():
            beamline.clean()

        # Run simulations, and update all the BeamParameters and LineParameters
        for (i, beamline) in enumerate(self._beamlines.values()):
            # TODO: implement multi-threading here
            beamline.generate_input(mapping)
            beamline.simulate(workers)
            beamline.update_watches_and_lines()

    def get_templates(self):
        """Get templates for all beamlines."""
        templates = []
        for beamline in self._beamlines.values():
            templates.append(beamline.template)
        return templates

    def __str__(self):
        text = ''
        for beamline in self._beamlines.values():
            text += beamline.__str__() + '\n'

        return text
