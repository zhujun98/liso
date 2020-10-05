"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import h5py


class SimWriter:
    """Write simulated data in HDF5 file."""

    def __init__(self, n_pulses, n_particles, path):
        """Initialization.

        :param int n_pulses: number of macro-pulses.
        :param int n_particles: number of particles per simulation.
        :param str path: path of the hdf5 file.
        """
        # TODO: restrict the number of data points in a single file
        self._n_pulses = n_pulses
        self._n_particles = n_particles

        self._path = path

        with h5py.File(path, 'w') as fp:
            fp.create_group('METADATA/SOURCE')
            fp.create_group('INDEX')
            fp.create_group('CONTROL')
            grp = fp.create_group('PHASESPACE')
            for axis in ['X', 'PX', 'Y', 'PY', 'Z', 'PZ', 'T']:
                grp.create_group(axis)

        self._initialized = False

    def write(self, idx, controls, phasespaces):
        """Write data into file incrementally.

        :param int idx: scan index.
        :param dict controls: dictionary of the control data.
        :param dict phasespaces: dictionary of the phasespace data.
        """
        with h5py.File(self._path, 'a') as fp:
            if not self._initialized:
                fp.create_dataset(
                    "INDEX/simId", (self._n_pulses,), dtype='i8')

                fp.create_dataset(
                    "METADATA/SOURCE/control", (len(controls),),
                    dtype=h5py.string_dtype())
                fp.create_dataset(
                    "METADATA/SOURCE/phasespace", (len(phasespaces),),
                    dtype=h5py.string_dtype())

                for i, k in enumerate(controls):
                    fp["METADATA/SOURCE/control"][i] = k
                    fp.create_dataset(
                        f"CONTROL/{k}", (self._n_pulses,), dtype='f8')

                for i, (k, v) in enumerate(phasespaces.items()):
                    fp["METADATA/SOURCE/phasespace"][i] = k
                    for col in v.columns:
                        fp.create_dataset(
                            f"PHASESPACE/{col.upper()}/{k}",
                            (self._n_pulses, self._n_particles),
                            dtype='f8')

                self._initialized = True

            fp["INDEX/simId"][idx] = idx + 1

            for k, v in controls.items():
                fp[f"CONTROL/{k}"][idx] = v

            for k, v in phasespaces.items():
                # TODO: the predefined number of particles must be exactly
                #       the same as the number in the data.
                for col in v.columns:
                    fp[f"PHASESPACE/{col.upper()}/{k}"][idx] = v[col]
