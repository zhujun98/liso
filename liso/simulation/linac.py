"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
from collections.abc import Mapping

from collections import OrderedDict
from .beamline import create_beamline
from .output import OutputData
from .input import generate_input, ParticleFileGenerator


class Linac(Mapping):
    """Linac class.

    The linac class is the abstraction of a linear accelerator. It
    consists of one or multiple beamlines. These beamlines can be
    simulated using different codes.
    """
    def __init__(self):
        self._generator = None
        self._beamlines = OrderedDict()

        self._n0 = None

    def __getitem__(self, item):
        """Override."""
        return self._beamlines[item]

    def __iter__(self):
        """Override."""
        return self._beamlines.__iter__()

    def __len__(self):
        """Override."""
        return self._beamlines.__len__()

    @property
    def n0(self):
        if self._n0 is None:
            raise RuntimeError("Call compile first to initialize the linac!")
        return self._n0

    def add_initial_distribution(self, *args, **kwargs):
        """Add initial particle distribution of the linac."""
        self._generator = ParticleFileGenerator()

    def add_beamline(self, code, *args, **kwargs):
        """Add a beamline.

        :param str code: Code used to simulate the beamline. ASTRA: 'astra'
            or 'a'; IMPACT-T: 'impactt' or 't'. Case Insensitive.
        """
        bl = create_beamline(code, *args, **kwargs)

        name = bl.name
        if name in self._beamlines:
            raise ValueError(f"Beamline {name} already exists!")

        lst = list(self._beamlines.keys())
        if len(lst) > 0:
            self._beamlines[lst[-1]].next = bl
        self._beamlines[name] = bl

    def add_watch(self, name, *args, **kwargs):
        """Add a Watch object to a beamline of the Linac.

        :param str name: beamline name.
        """
        try:
            bl = self._beamlines[name]
        except KeyError:
            raise KeyError(f"Beamline {name} does not exist!")

        bl.add_watch(*args, **kwargs)

    def _check_template(self, mapping):
        templates = []
        for bl in self._beamlines.values():
            templates.append(bl._template)

        found = set()
        for template in templates:
            found = found.union(
                generate_input(template, mapping, dry_run=True))

        not_found = mapping.keys() - found
        if not_found:
            raise ValueError(f"{not_found} not found in the templates!")

    def compile(self):
        """Initialize the linac."""
        bl0 = self._beamlines[next(iter(self._beamlines))]

        self._n0 = bl0.n0

    def run(self, mapping, *, n_workers=1, timeout=None):
        """Run simulation for all the beamlines.

        :param dict mapping:
        :param int n_workers: Number of processes used in simulation.
        :param float timeout: Maximum allowed duration in seconds of the
            simulation.
        """
        if self._generator is not None:
            pass

        self._check_template(mapping)
        for i, bl in enumerate(self._beamlines.values()):
            bl.run(mapping, n_workers, timeout)

    async def async_run(self, idx, mapping, tmp_dir, *, timeout=None):
        self._check_template(mapping)

        inputs = dict()
        phasespaces = dict()
        for i, bl in enumerate(self._beamlines.values()):
            output = await bl.async_run(mapping, tmp_dir, timeout=timeout)
            inputs.update(output['input'])
            phasespaces.update(output['phasespace'])
        return idx, OutputData(inputs, phasespaces)

    def status(self):
        """Return the status of the linac."""
        ret = OrderedDict()
        for name, bl in self._beamlines.items():
            ret[name] = bl.status()
        return ret

    def __str__(self):
        text = '\n' + '=' * 80 + '\n'
        text += 'Linac definition:\n\n'
        for bl in self._beamlines.values():
            text += bl.__str__()
        text += '=' * 80 + '\n'
        return text
