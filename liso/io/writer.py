"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import abc
from datetime import datetime

import h5py

from ..proc import Phasespace


class _BaseWriter(abc.ABC):
    """Base class for HDF5 writer."""

    # cap the number of sequence files
    _MAX_SEQUENCE = 99

    def __init__(self, path, *,
                 chunk_size=50,
                 max_events_per_file=100000):
        """Initialization.

        :param str path: path of the hdf5 file.
        :param int chunk_size: size of the first dimention of a chunk in
            a dataset.
        :param int max_events_per_file: maximum events stored in a single file.

        :raise OSError is the file already exists.
        """
        if max_events_per_file < chunk_size:
            raise ValueError(
                "max_events_per_file cannot be smaller than chunk_size")
        if max_events_per_file < 500:
            raise ValueError("max_events_per_file must be at least 500!")
        self._chunk_size = chunk_size
        self._max_events_per_file = max_events_per_file

        # not allow to overwrite existing file
        fp = h5py.File(path, 'w-')
        fp.create_dataset("METADATA/createDate",
                          data=datetime.now().isoformat())
        fp.create_dataset("METADATA/updateDate",
                          data=datetime.now().isoformat())

        self._fp = fp

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._finalize()
        self.close()

    @abc.abstractmethod
    def _finalize(self):
        pass

    def close(self):
        self._fp.close()


class SimWriter(_BaseWriter):
    """Write simulated data in HDF5 file."""

    def __init__(self, path, *, start_id=1, schema, **kwargs):
        """Initialization.

        :param str path: path of the hdf5 file.
        :param int start_id: starting simulation id.
        :param tuple schema: (control, phasespace) data schema
        """
        super().__init__(path, **kwargs)

        if not isinstance(start_id, int) or start_id < 1:
            raise ValueError(
                f"start_id must a positive integer. Actual: {start_id}")
        self._start_id = start_id
        self._sim_ids = []

        self._control_schema, self._phasespace_schema = schema

        self._init_channel_data("control", self._control_schema)
        self._init_channel_data("phasespace", self._phasespace_schema)

    def _init_channel_data(self, channel_category, schema):
        fp = self._fp

        meta_ch = f"METADATA/{channel_category}Channel"
        fp.create_dataset(meta_ch,
                          dtype=h5py.string_dtype(),
                          shape=(len(schema),))

        for i, (k, v) in enumerate(schema.items()):
            fp[meta_ch][i] = k
            dtype = v['type']
            if dtype == 'phasespace':
                for col in Phasespace.columns:
                    fp.create_dataset(
                        f"{channel_category.upper()}/{col.upper()}/{k}",
                        shape=(self._chunk_size, v['macroparticles']),
                        dtype='<f8',
                        chunks=(self._chunk_size, v['macroparticles']),
                        maxshape=(self._max_events_per_file, v['macroparticles']))
            else:
                fp.create_dataset(
                    f"{channel_category.upper()}/{k}",
                    shape=(self._chunk_size,),
                    dtype=v['type'],
                    chunks=(self._chunk_size,),
                    maxshape=(self._max_events_per_file,))

    def write(self, idx, controls, phasespaces):
        """Write data from one simulation into the file.

        :param int idx: scan index, starting from 0.
        :param dict controls: dictionary of the control data.
        :param dict phasespaces: dictionary of the phasespace data.
        """
        fp = self._fp
        chunk_size = self._chunk_size

        if idx > 0 and idx % chunk_size == 0:
            n_chunks = idx // chunk_size + 1
            for k in controls:
                fp[f"CONTROL/{k}"].resize(n_chunks * chunk_size, axis=0)

            for k in phasespaces:
                for col in Phasespace.columns:
                    fp[f"PHASESPACE/{col.upper()}/{k}"].resize(
                        n_chunks * chunk_size, axis=0)

        for k, v in controls.items():
            fp[f"CONTROL/{k}"][idx] = v

        for k, v in phasespaces.items():
            try:
                # The rational behind writing different columns separately
                # is to avoid reading out all the columns when only one
                # or two columns are needed.
                for col in v.columns:
                    fp[f"PHASESPACE/{col.upper()}/{k}"][idx] = v[col]
            except TypeError:
                # particle loss
                pass

        self._sim_ids.append(idx + self._start_id)

    def _finalize(self):
        """Override."""
        fp = self._fp
        # Caveat: simulation results do not arrive in order
        fp.create_dataset("INDEX/simId",
                          data=sorted(self._sim_ids), dtype='u8')
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
                    maxshape=(self._max_events_per_file, *v['shape']))
            else:
                fp.create_dataset(
                    f"{channel_category.upper()}/{k}",
                    shape=(self._chunk_size,),
                    dtype=v['type'],
                    chunks=(self._chunk_size,),
                    maxshape=(self._max_events_per_file,))

    def write(self, pulse_id, controls, instruments):
        """Write matched data from one train into the file.

        :param int pulse_id: macro-pulse ID.
        :param dict controls: dictionary of the control data.
        :param dict instruments: dictionary of the phasespace data.
        """
        fp = self._fp
        idx = len(self._pulse_ids) % self._max_events_per_file

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
