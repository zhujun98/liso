"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
from datetime import datetime

import h5py


class _BaseWriter:
    """Base class for HDF5 writer."""
    def __init__(self, path, *, chunk_size=50, max_size_per_file=5000):
        """Initialization.

        :param str path: path of the hdf5 file.
        :param int chunk_size: size of the first dimention of a chunk in
            a dataset.
        :param int max_size_per_file: maximum size of a dataset.

        :raise OSError is the file already exists.
        """
        # not allow to overwrite existing file
        fp = h5py.File(path, 'w-')
        fp.create_dataset("METADATA/createDate",
                          data=datetime.now().isoformat())
        fp.create_dataset("METADATA/updateDate",
                          data=datetime.now().isoformat())

        self._fp = fp

        self._chunk_size = chunk_size
        self._max_size_per_file = max_size_per_file

        self._initialized = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._finalize()
        self.close()

    def _finalize(self):
        ...

    def close(self):
        self._fp.close()


class SimWriter(_BaseWriter):
    """Write simulated data in HDF5 file."""

    def __init__(self, n_pulses, n_particles, path, *,
                 start_id=1, schema=None, **kwargs):
        """Initialization.

        :param int n_pulses: number of macro-pulses.
        :param int n_particles: number of particles per simulation.
        :param str path: path of the hdf5 file.
        :param int start_id: starting simulation id.
        :param tuple schema: (control, phasespace) data schema
        """
        super().__init__(path, **kwargs)

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
        fp = self._fp
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


class ExpWriter(_BaseWriter):

    def __init__(self, path, *, schema, **kwargs):
        """Initialization.

        :param str path: path of the hdf5 file.
        :param tuple schema: (control, instrument) data schema.
        """
        super().__init__(path, **kwargs)

        self._control_schema, self._instrument_schema = schema

        self._init_channel_data("control", self._control_schema)
        self._init_channel_data("instrument", self._instrument_schema)

        self._pulse_ids = []

    def _init_channel_data(self, channel_category, schema):
        fp = self._fp

        meta_ch = f"METADATA/{channel_category}Channel"
        fp.create_dataset(meta_ch,
                          dtype=h5py.string_dtype(),
                          shape=(len(schema),))

        for i, (k, v) in enumerate(schema.items()):
            fp[meta_ch][i] = k
            dtype = v['type']
            if dtype == 'NDArray':
                fp.create_dataset(
                    f"{channel_category.upper()}/{k}",
                    shape=(self._chunk_size, *v['shape']),
                    dtype=v['dtype'],
                    chunks=(self._chunk_size, *v['shape']),
                    maxshape=(self._max_size_per_file, *v['shape']))
            else:
                fp.create_dataset(
                    f"{channel_category.upper()}/{k}",
                    shape=(self._chunk_size,),
                    dtype=v['type'],
                    chunks=(self._chunk_size,),
                    maxshape=(self._max_size_per_file,))

    def write(self, pulse_id, controls, instruments):
        """Write matched data from one train into the file.

        :param int pulse_id: macro-pulse ID.
        :param dict controls: dictionary of the control data.
        :param dict instruments: dictionary of the phasespace data.
        """
        fp = self._fp
        idx = len(self._pulse_ids) % self._max_size_per_file

        for k, v in controls.items():
            fp[f"CONTROL/{k}"][idx] = v
        for k, v in instruments.items():
            fp[f"INSTRUMENT/{k}"][idx] = v

        self._pulse_ids.append(pulse_id)

    def _finalize(self):
        """Override."""
        fp = self._fp
        fp.create_dataset("INDEX/pulseId", data=self._pulse_ids, dtype='u8')
        fp["METADATA/updateDate"][()] = datetime.now().isoformat()
