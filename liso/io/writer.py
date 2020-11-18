"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import abc
from datetime import datetime
import os
import pathlib
from string import Template

import h5py

from ..proc import Phasespace


class _BaseWriter(abc.ABC):
    """Base class for HDF5 writer."""

    # cap the number of sequence files
    _MAX_SEQUENCE = 99

    _IMAGE_CHUNK = (16, 512)

    def __init__(self, path, *,
                 chmod=True,
                 group=1,
                 chunk_size=50,
                 max_events_per_file):
        """Initialization.

        :param pathlib.Path path: path of the simulation/run folder.
        :param bool chmod: True for changing the permission to 400 after
            finishing writing.
        :param int group: writer group (1-99).
        :param int chunk_size: size of the first dimention of a chunk in
            a dataset.
        :param int max_events_per_file: maximum events stored in a single file.

        :raise OSError is the file already exists.
        """
        if max_events_per_file < chunk_size:
            raise ValueError(
                "max_events_per_file cannot be smaller than chunk_size")

        self._chunk_size = chunk_size
        self._max_events_per_file = max_events_per_file

        self._path = path if isinstance(path, pathlib.Path) \
            else pathlib.Path(path)
        self._fp = None

        self._chmod = chmod

        self._group = group
        if not isinstance(group, int) or group < 1 or group > 99:
            raise ValueError("group must be an integer within [1, 99]")

        self._index = 0
        self._file_count = 0

    @abc.abstractmethod
    def _create_new_file(self):
        """Create a new file in the run folder."""
        pass

    def _init_meta_data(self):
        fp = self._fp
        fp.create_dataset("METADATA/createDate",
                          data=datetime.now().isoformat())
        fp.create_dataset("METADATA/updateDate",
                          data=datetime.now().isoformat())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @abc.abstractmethod
    def _finalize(self):
        pass

    def close(self):
        if self._fp is not None:
            self._finalize()
            filename = self._fp.filename
            self._fp.close()
            if self._chmod:
                os.chmod(filename, 0o400)
            self._fp = None


class SimWriter(_BaseWriter):
    """Write simulated data in HDF5 file."""

    _FILE_ROOT_NAME = Template("SIM-G$group-S$seq.hdf5")

    def __init__(self, path, *, schema, max_events_per_file=10000, **kwargs):
        """Initialization.

        :param str/pathlib.Path path: path of the simulation folder.
        :param tuple schema: (control, phasespace) data schema
        """
        super().__init__(path,
                         max_events_per_file=max_events_per_file, **kwargs)

        self._sim_ids = []

        self._control_schema, self._phasespace_schema = schema

    def _create_new_file(self):
        """Override."""
        next_file = self._path.joinpath(self._FILE_ROOT_NAME.substitute(
            group=f"{self._group:02d}", seq=f"{self._file_count:06d}"))
        self._file_count += 1
        self._sim_ids.clear()

        self._fp = h5py.File(next_file, 'w-')

        self._init_meta_data()
        self._init_channel_data("control", self._control_schema)
        self._init_channel_data("phasespace", self._phasespace_schema)

        return self._fp

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

    def write(self, sim_id, controls, phasespaces):
        """Write data from one simulation into the file.

        :param int sim_id: simulation ID.
        :param dict controls: dictionary of the control data.
        :param dict phasespaces: dictionary of the phasespace data.
        """
        fp = self._fp
        chunk_size = self._chunk_size
        idx = self._index

        if idx % self._max_events_per_file == 0:
            # close the current file
            self.close()
            # create a new one
            fp = self._create_new_file()
            self._index = idx = 0
        elif idx % chunk_size == 0:
            n_chunks = idx // chunk_size + 1
            new_size = min(n_chunks * chunk_size, self._max_events_per_file)
            for k in self._control_schema:
                fp[f"CONTROL/{k}"].resize(new_size, axis=0)
            for k in self._phasespace_schema:
                for col in Phasespace.columns:
                    fp[f"PHASESPACE/{col.upper()}/{k}"].resize(new_size, axis=0)

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

        # Caveat: sim ID does not arrive in sequence
        self._sim_ids.append(sim_id)
        self._index += 1

    def _finalize(self):
        """Override."""
        fp = self._fp
        fp.create_dataset("INDEX/simId", data=self._sim_ids, dtype='u8')
        fp["METADATA/updateDate"][()] = datetime.now().isoformat()


class ExpWriter(_BaseWriter):

    _FILE_ROOT_NAME = Template("RAW-$run-G$group-S$seq.hdf5")

    def __init__(self, path, *, schema, max_events_per_file=500, **kwargs):
        """Initialization.

        :param pathlib.Path path: path of the run folder.
        :param tuple schema: (control, diagnostic) data schema.
        """
        super().__init__(path,
                         max_events_per_file=max_events_per_file, **kwargs)

        self._control_schema, self._diagnostic_schema = schema

        self._pulse_ids = []

    def _create_new_file(self):
        """Override."""
        next_file = self._path.joinpath(self._FILE_ROOT_NAME.substitute(
            run=self._path.name.upper(),
            group=f"{self._group:02d}",
            seq=f"{self._file_count:06d}"))
        self._file_count += 1
        self._pulse_ids.clear()

        self._fp = h5py.File(next_file, 'w-')

        self._init_meta_data()
        self._init_channel_data("control", self._control_schema)
        self._init_channel_data("diagnostic", self._diagnostic_schema)

        return self._fp

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
                shape = v['shape']
                if len(shape) == 2:
                    # image data
                    chunk_size = (1, *self._IMAGE_CHUNK)
                else:
                    # TODO: finish it
                    chunk_size = (self._chunk_size, *v['shape'])
                fp.create_dataset(
                    f"{channel_category.upper()}/{k}",
                    shape=(self._chunk_size, *v['shape']),
                    dtype=v['dtype'],
                    chunks=chunk_size,
                    maxshape=(self._max_events_per_file, *v['shape']))
            else:
                fp.create_dataset(
                    f"{channel_category.upper()}/{k}",
                    shape=(self._chunk_size,),
                    dtype=v['type'],
                    chunks=(self._chunk_size,),
                    maxshape=(self._max_events_per_file,))

    def write(self, pulse_id, controls, diagnostics):
        """Write matched data from one train into the file.

        :param int pulse_id: macro-pulse ID.
        :param dict controls: dictionary of the control data.
        :param dict diagnostics: dictionary of the phasespace data.
        """
        fp = self._fp
        chunk_size = self._chunk_size
        idx = self._index

        if idx % self._max_events_per_file == 0:
            # close the current file
            self.close()
            # create a new one
            fp = self._create_new_file()
            self._index = idx = 0
        elif idx % chunk_size == 0:
            n_chunks = idx // chunk_size + 1
            new_size = min(n_chunks * chunk_size, self._max_events_per_file)
            for k in self._control_schema:
                fp[f"CONTROL/{k}"].resize(new_size, axis=0)
            for k in self._diagnostic_schema:
                fp[f"DIAGNOSTIC/{k}"].resize(new_size, axis=0)

        for k, v in controls.items():
            fp[f"CONTROL/{k}"][idx] = v
        for k, v in diagnostics.items():
            fp[f"DIAGNOSTIC/{k}"][idx] = v

        self._pulse_ids.append(pulse_id)
        self._index += 1

    def _finalize(self):
        """Override."""
        fp = self._fp
        fp.create_dataset("INDEX/pulseId", data=self._pulse_ids, dtype='u8')
        fp["METADATA/updateDate"][()] = datetime.now().isoformat()
