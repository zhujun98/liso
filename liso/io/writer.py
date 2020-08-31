"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import h5py


class SimWriter:
    """Write simulation parameters in file."""

    def __init__(self, pulses, particles, path):
        """Initialization.

        :param int pulses: number of macro-pulses.
        :param int particles: number of particles per simulation.
        :param str path: path of the hdf5 file.
        """
        # TODO: restrict the number of data points in a single file
        self._n_pulses = pulses
        self._n_particles = particles

        self._path = path
        # self._file.attrs['writer'] = f'liso {__version__}'

        with h5py.File(path, 'w') as fp:
            fp.create_group('metadata')

            fp.create_group('input')

            # phasespace
            grp = fp.create_group('phasespace')
            for axis in ['x', 'px', 'y', 'py', 'z', 'pz', 't']:
                grp.create_group(axis)

        self._initialized = False

    def write(self, idx, data):
        """Write data into file incrementally.

        :param int idx: scan index.
        :param OutputData data: output data.
        """
        with h5py.File(self._path, 'a') as fp:
            if not self._initialized:
                for k in data['metadata']:
                    ds = fp.create_dataset(f"metadata/{k}",
                                           (len(data['metadata'][k]), ),
                                           dtype=h5py.string_dtype())
                    ds[:] = list(data['metadata'][k])

                for k in data['input']:
                    fp.create_dataset(
                        f"input/{k}", (self._n_pulses,), dtype='f8')

                for k, v in data['phasespace'].items():
                    for col in v.columns:
                        fp.create_dataset(f"phasespace/{col}/{k}",
                                          (self._n_pulses, self._n_particles),
                                          dtype='f8')

                self._initialized = True

            for k, v in data['input'].items():
                fp[f"input/{k}"][idx] = v

            for k, v in data['phasespace'].items():
                for col in v.columns:
                    fp[f"phasespace/{col}/{k}"][idx] = v[col]
