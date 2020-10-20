"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
from datetime import datetime

import h5py


class _BaseWriter:
    """Base class for HDF5 writer."""
    def __init__(self, path):
        """Initialization.

        :param str path: path of the hdf5 file.
        """
        self._path = path

        # not allow to overwrite existing file
        with h5py.File(self._path, 'a') as fp:
            fp.create_dataset("METADATA/createDate",
                              data=datetime.now().isoformat())
            fp.create_dataset("METADATA/updateDate",
                              data=datetime.now().isoformat())
        self._initialized = False


class SimWriter(_BaseWriter):
    """Write simulated data in HDF5 file."""

    def __init__(self, n_pulses, n_particles, path, *, start_id=1):
        """Initialization.

        :param int n_pulses: number of macro-pulses.
        :param int n_particles: number of particles per simulation.
        :param str path: path of the hdf5 file.
        :param int start_id: starting simulation id.
        """
        super().__init__(path)

        self._n_pulses = n_pulses
        self._n_particles = n_particles

        if not isinstance(start_id, int) or start_id < 1:
            raise ValueError(
                f"start_id must a positive integer. Actual: {start_id}")
        self._start_id = start_id

    def write(self, idx, controls, phasespaces):
        """Write data from one simulation into the file.

        :param int idx: scan index.
        :param dict controls: dictionary of the control data.
        :param dict phasespaces: dictionary of the phasespace data.
        """
        with h5py.File(self._path, 'a') as fp:
            if not self._initialized:
                fp.create_dataset(
                    "METADATA/controlChannel", (len(controls),),
                    dtype=h5py.string_dtype())
                fp.create_dataset(
                    "METADATA/phasespaceChannel", (len(phasespaces),),
                    dtype=h5py.string_dtype())

                fp.create_dataset(
                    "INDEX/simId", (self._n_pulses,), dtype='u8')

                for i, k in enumerate(controls):
                    fp["METADATA/controlChannel"][i] = k
                    fp.create_dataset(
                        f"CONTROL/{k}", (self._n_pulses,), dtype='f8')

                for i, (k, v) in enumerate(phasespaces.items()):
                    fp["METADATA/phasespaceChannel"][i] = k
                    for col in v.columns:
                        fp.create_dataset(
                            f"PHASESPACE/{col.upper()}/{k}",
                            (self._n_pulses, self._n_particles),
                            dtype='f8')

                self._initialized = True

            fp["INDEX/simId"][idx] = idx + self._start_id

            for k, v in controls.items():
                fp[f"CONTROL/{k}"][idx] = v

            for k, v in phasespaces.items():
                if len(v) == self._n_particles:
                    # The rational behind writing different columns separately
                    # is to avoid reading out all the columns when only one
                    # or two columns are needed.
                    for col in v.columns:
                        fp[f"PHASESPACE/{col.upper()}/{k}"][idx] = v[col]

            fp["METADATA/updateDate"][()] = datetime.now().isoformat()
