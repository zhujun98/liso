"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
from collections.abc import Mapping

from collections import defaultdict, OrderedDict
from .beamline import create_beamline


class Linac(Mapping):
    """Linac class.

    The linac class is the abstraction of a linear accelerator. It
    consists of one or multiple beamlines. These beamlines can be
    simulated using different codes.
    """
    def __init__(self, mps=None):
        """Initialization.

        :param int/None mps: number of macro-particles at the
            start of the simulation.
        """
        self._mps = mps

        self._beamlines = OrderedDict()

    def __getitem__(self, item):
        """Override."""
        return self._beamlines[item]

    def __iter__(self):
        """Override."""
        return self._beamlines.__iter__()

    def __len__(self):
        """Override."""
        return self._beamlines.__len__()

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

    @property
    def schema(self):
        """Return the schema of phasespace data."""
        if self._mps is None:
            raise ValueError("Please specify the number of particles "
                             "when instantiating a Linac object!")

        phasespace_schema = dict()
        for name in self._beamlines:
            phasespace_schema[f"{name}/out"] = {
                "macroparticles": self._mps,
                "type": "phasespace",
            }
        return phasespace_schema

    def _split_mapping(self, mapping):
        """Split mapping into different groups.

        The keys in the dictionary "mapping" are expected to have the
        format beamline/variable.
        """
        mapping_grp = defaultdict(dict)
        first_bl = next(iter(self._beamlines))
        for key, value in mapping.items():
            splitted = key.split('/', 1)
            if len(splitted) == 1:
                mapping_grp[first_bl][splitted[0]] = value
            else:
                mapping_grp[splitted[0]][splitted[1]] = value
        return mapping_grp

    def compile(self, mapping):
        """Compile all the inputs before running the simulation."""
        mapping_grp = self._split_mapping(mapping)
        mapping_norm = {}
        for name, bl in self._beamlines.items():
            bl.compile(mapping_grp[name])
            mapping_norm.update({
                f"{name}/{k}": v for k, v in mapping_grp[name].items()
            })
        return mapping_norm

    def run(self, mapping, *, n_workers=1, timeout=None):
        """Run simulation for all the beamlines.

        :param dict mapping:
        :param int n_workers: Number of processes used in simulation.
        :param float timeout: Maximum allowed duration in seconds of the
            simulation.
        """
        self.compile(mapping)

        out = None
        for i, bl in enumerate(self._beamlines.values()):
            out = bl.run(out, timeout=timeout, n_workers=n_workers)

    async def async_run(self, sim_id, mapping, *, timeout=None):
        controls = self.compile(mapping)

        out = None
        phasespaces = OrderedDict()
        for name, bl in self._beamlines.items():
            out = await bl.async_run(out, f'tmp{sim_id:06d}', timeout=timeout)
            phasespaces[f"{name}/out"] = out
        return sim_id, controls, phasespaces

    def status(self):
        """Return the status of the linac."""
        ret = OrderedDict()
        for name, bl in self._beamlines.items():
            ret[name] = bl.status()
        return ret

    def __str__(self):
        text = '\n' + '=' * 80 + '\n'
        text += 'Linac definition:\n'
        for bl in self._beamlines.values():
            text += '\n'
            text += bl.__str__()
        text += '=' * 80 + '\n'
        return text
