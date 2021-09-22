"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
from collections.abc import Mapping
from collections import defaultdict, OrderedDict
from typing import Optional, Tuple

from ..exceptions import LisoRuntimeError
from .beamline import create_beamline


class Linac(Mapping):
    """A linear accelerator complex for simulation.

    The linac class is the abstraction of a linear accelerator. It
    consists of one or multiple beamlines. These beamlines can be
    simulated using different codes.
    """
    def __init__(self, mps: Optional[int] = None):
        """Initialization.

        :param mps: Number of macro-particles at the start of the simulation.
        """
        self._mps = mps  # Number of macro-particles.

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

    def add_beamline(self, code: str, *args, **kwargs) -> None:
        """Add a beamline.

        The args and kwargs will be passed to the constructor of the
        corresponding :class:`liso.simulation.beamline.Beamline` type.

        :param code: Simulation code name (case insensitive) used to
            simulate the beamline: 'astra' or 'a' for ASTRA; 'impactt' or
            't' for IMPACT-T; 'elegant' or 'e' for ELEGANT.
        """
        bl = create_beamline(code, *args, **kwargs)

        name = bl.name
        if name in self._beamlines:
            raise ValueError(f"Beamline {name} already exists!")

        lst = list(self._beamlines.keys())
        if len(lst) > 0:
            # not the first Beamline
            self._beamlines[lst[-1]].next = bl
        self._beamlines[name] = bl

    def add_watch(self, name: str, *args, **kwargs):
        """Add a Watch object to a beamline of the Linac.

        :param name: beamline name.
        """
        try:
            bl = self._beamlines[name]
        except KeyError as e:
            raise KeyError(f"Beamline {name} does not exist!") from e

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

    def check_temp_swd(self, start_id, end_id):
        """Check whether temporary simulation directories already exist.

        :raises FileExistsError: If there is already any directory which has
            the same name as the temporary simulation directory to be created.
        """
        for _, bl in self._beamlines.items():
            bl.check_temp_swd(start_id, end_id)

    def run(self, params: dict, *,
            n_workers: int = 1, timeout: Optional[int] = None) -> None:
        """Run simulation for all the beamlines.

        :param params: A dictionary of parameters used in the simulation
            input file.
        :param n_workers: Number of processes used in simulation.
        :param timeout: Maximum allowed duration in seconds of thesimulation.
        """
        self.compile(params)

        out = None
        for bl in self._beamlines.values():
            out = bl.run(out, timeout=timeout, n_workers=n_workers)

    async def async_run(self, sim_id: int, params: dict, *,
                        timeout=None) -> Tuple[int, dict]:
        """Run simulation for all the beamlines asynchronously.

        :param sim_id: Id of the current simulation.
        :param params: A dictionary of parameters used in the simulation
            input file.
        :param timeout: Maximum allowed duration in seconds of the
            simulation.

        :raises LisoRuntimeError: If any Beamline cannot create a
            temporary directory to run simulation.
        """
        controls = self.compile(params)

        out = None
        phasespaces = OrderedDict()
        for name, bl in self._beamlines.items():
            try:
                out = await bl.async_run(sim_id, out, timeout=timeout)
            except FileExistsError as e:
                raise LisoRuntimeError(
                    "Failed to create temporary simulation directory!") from e
            phasespaces[f"{name}/out"] = out
        return sim_id, {'control': controls, 'phasespace': phasespaces}

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
