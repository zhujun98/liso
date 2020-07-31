"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import h5py

from .. import __version__


class SimParamWriter:
    """Write simulation parameters in file."""

    def __init__(self, path, params):
        self._file = h5py.File(path, 'w')
        self.params = params
        self.indexes = {}  # {path: (first, count)}

    def prepare_source(self, source):
        """Prepare all the datasets for one source.
        We do this as a separate step so the contents of the file are defined
        together before the main data.
        """
        for key in sorted(self.data.keys_for_source(source)):
            path = f"{self._section(source)}/{source}/{key.replace('.', '/')}"
            nentries = self._guess_number_of_storing_entries(source, key)
            src_ds1 = self.data._source_index[source][0].file[path]
            self.file.create_dataset_like(
                path, src_ds1, shape=(nentries,) + src_ds1.shape[1:],
                # Corrected detector data has maxshape==shape, but if any max
                # dim is smaller than the chunk size, h5py complains. Making
                # the first dimension unlimited avoids this.
                maxshape=(None,) + src_ds1.shape[1:],
            )
            if source in self.data.instrument_sources:
                self.data_sources.add(f"INSTRUMENT/{source}/{key.partition('.')[0]}")

        if source not in self.data.instrument_sources:
            self.data_sources.add(f"CONTROL/{source}")

    def copy_source(self, source):
        """Copy data for all keys of one source"""
        for key in self.data.keys_for_source(source):
            self.copy_dataset(source, key)

    def write_indexes(self):
        """Write the INDEX information for all data we've copied"""
        for groupname, (first, count) in self.indexes.items():
            group = self.file.create_group(f'INDEX/{groupname}')
            group.create_dataset('first', data=first, dtype=np.uint64)
            group.create_dataset('count', data=count, dtype=np.uint64)

    def write_metadata(self):
        """Write the METADATA section, including lists of sources"""
        vlen_bytes = h5py.special_dtype(vlen=bytes)
        data_sources = sorted(self.data_sources)
        N = len(data_sources)

        sources_ds = self.file.create_dataset(
            'METADATA/dataSourceId', (N,), dtype=vlen_bytes, maxshape=(None,)
        )
        sources_ds[:] = data_sources

        root_ds = self.file.create_dataset(
            'METADATA/root', (N,), dtype=vlen_bytes, maxshape=(None,)
        )
        root_ds[:] = [ds.split('/', 1)[0] for ds in data_sources]

        devices_ds = self.file.create_dataset(
            'METADATA/deviceId', (N,), dtype=vlen_bytes, maxshape=(None,)
        )
        devices_ds[:] = [ds.split('/', 1)[1] for ds in data_sources]

    def write(self):
        d = self.data

        # set writer version
        self.file.attrs['writer'] = 'liso {}'.format(__version__)

        self.file.create_dataset(
            'INDEX/trainId', data=d.train_ids, dtype='u8'
        )

        for source in d.all_sources:
            self.prepare_source(source)

        self.write_metadata()

        for source in d.all_sources:
            self.copy_source(source)

        self.write_indexes()
