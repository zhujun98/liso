"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import h5py

from .. import __version__


class SimParamWriter:
    """Write simulation parameters in file."""

    def __init__(self, path, data):
        self._file = h5py.File(path, 'w')
        self._data = data

    def write_metadata(self):
        """Write the METADATA section, including lists of sources"""
        vlen_bytes = h5py.special_dtype(vlen=bytes)
        data_sources = sorted(self.data_sources)
        N = len(data_sources)

        sources_ds = self._file.create_dataset(
            'METADATA/dataSourceId', (N,), dtype=vlen_bytes, maxshape=(None,)
        )
        sources_ds[:] = data_sources

        root_ds = self._file.create_dataset(
            'METADATA/root', (N,), dtype=vlen_bytes, maxshape=(None,)
        )
        root_ds[:] = [ds.split('/', 1)[0] for ds in data_sources]

        devices_ds = self._file.create_dataset(
            'METADATA/deviceId', (N,), dtype=vlen_bytes, maxshape=(None,)
        )
        devices_ds[:] = [ds.split('/', 1)[1] for ds in data_sources]

    def write(self):
        d = self._data

        # set writer version
        self._file.attrs['writer'] = f'liso {__version__}'

        self._file.create_dataset(
            'INDEX/trainId', data=d.train_ids, dtype='u8'
        )

        self.write_metadata()
