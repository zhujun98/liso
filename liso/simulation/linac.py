"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
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

        name = bl.name
        if name in self._beamlines:
            raise ValueError(f"Beamline {name} already exists!")

        # connect the new Beamline to the last added Beamline
        if len(self._beamlines) != 0:
            list(self._beamlines.values())[-1].next = bl

            if bl.pin is None:
                raise ValueError("Initial particle file of the new Beamline must "
                                 "be specified for a concatenated simulation!")

        self._beamlines[name] = bl

    def run(self, mapping, *, n_workers=1, timeout=600):
        """Run simulation for the beamline

        :param int n_workers: Number of processes used in simulation.
        :param float timeout: Maximum allowed duration in seconds of the
            simulation.
        """
        # First clean all the previous output
        for beamline in self._beamlines.values():
            beamline.clean()

        # Run simulations, and update all the BeamParameters and LineParameters
        for i, bl in enumerate(self._beamlines.values()):
            bl.generate_input(mapping)
            bl.simulate(n_workers, timeout)
            bl.update_out()
            bl.update_watches_and_lines()

    def simulate(self, mapping):
        """Simulate and update all BeamParameters and LineParameters.

        Note:
        Even without space-charge effects, the code is ASTRA
        (or other codes)-bound. The Beamline method `simulation` takes
        most of the time.

        :param dict mapping: A dictionary for variables and covariables -
                             {name: value}.
        """
        # First clean all the previous output
        for beamline in self._beamlines.values():
            beamline.clean()

        # Run simulations, and update all the BeamParameters and LineParameters
        for i, bl in enumerate(self._beamlines.values()):
            bl.generate_input(mapping)
            bl.simulate()
            bl.update_out()
            bl.update_watches_and_lines()

    def __str__(self):
        text = ''
        for i, beamline in enumerate(self._beamlines.values()):
            text += "\nBeamline {:02d}\n".format(i)
            text += "-"*11 + "\n"
            text += beamline.__str__()
        return text
