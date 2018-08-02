#!/usr/bin/python
"""
Author: Jun Zhu, zhujun981661@gmail.com
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

        :param str code: Code used to simulate the beamline. ASTRA: 'astra'
            or 'a'; IMPACT-T: 'impactt' or 't'; IMPACT-Z: 'impactz' or 'z';
            GENESIS: 'genesis' or 'g'. Case Insensitive.
        :param \*args: Pass to the __init__ method of Beamline.
        :param \*\*kwargs: Pass to the __init__ method of Beamline.
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

        self._beamlines[bl.name] = bl

    def add_watch(self, beamline, *args, **kwargs):
        """Add a Watch object to a Beamline of the Linac.

        :param str beamline: Name of the Beamline object.
        :param \*args: Pass to the __init__ method of Watch.
        :param \*\*kwargs: Pass to the __init__ method of Watch.
        """
        self._beamlines[beamline].add_watch(*args, **kwargs)

    def simulate(self, mapping, workers=1):
        """Simulate and update all BeamParameters and LineParameters.

        Note:
        Even without space-charge effects, the code is ASTRA
        (or other codes)-bound. The Beamline method `simulation` takes
        most of the time.

        :param dict mapping: A dictionary for variables and covariables -
                             {name: value}.
        :param int workers: Number of threads.
        """
        # First clean all the previous output
        for beamline in self._beamlines.values():
            beamline.clean()

        # Run simulations, and update all the BeamParameters and LineParameters
        for i, beamline in enumerate(self._beamlines.values()):
            beamline.generate_input(mapping)
            beamline.simulate(workers)
            beamline.update_out()
            beamline.update_watches_and_lines()

    def _get_templates(self):
        """Get templates for all beamlines."""
        templates = []
        for beamline in self._beamlines.values():
            templates.append(beamline.template)
        return templates

    def __str__(self):
        text = ''
        for i, beamline in enumerate(self._beamlines.values()):
            text += "\nBeamline {:02d}\n".format(i)
            text += "-"*11 + "\n"
            text += beamline.__str__()
        return text
