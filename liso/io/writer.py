"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import h5py


class SimWriter:
    """Write simulation parameters in file."""

    # maximum number of data points
    _MAX_LEN = 10000

    def __init__(self, path):
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
            max_len = self._MAX_LEN
            if not self._initialized:
                for k in data['metadata']:
                    ds = fp.create_dataset(f"metadata/{k}",
                                           (len(data['metadata'][k]), ),
                                           dtype=h5py.string_dtype())
                    ds[:] = list(data['metadata'][k])

                for k in data['input']:
                    fp.create_dataset(f"input/{k}", (max_len,), dtype='f8')

                for k, v in data['phasespace'].items():
                    for col in v.columns:
                        # TODO: how to pass the number of particles information?
                        fp.create_dataset(f"phasespace/{col}/{k}",
                                          (max_len, 2000),
                                          dtype='f8')

                self._initialized = True

            for k, v in data['input'].items():
                fp[f"input/{k}"][idx] = v

            for k, v in data['phasespace'].items():
                for col in v.columns:
                    fp[f"phasespace/{col}/{k}"][idx] = v[col]
