"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import abc
from datetime import datetime
import os
from pathlib import Path
import re
from string import Template
from typing import Union

import h5py

from ..proc import Phasespace


def create_next_run_folder(parent: Union[str, Path], sim: bool = False) -> Path:
    """Create and return the next run folder to store the output data."""
    parent = Path(parent)
    # It is allowed to use an existing parent directory,
    # but not a run folder.
    parent.mkdir(exist_ok=True)

    start = 's' if sim else 'r'
    next_run_index = 1  # starting from 1
    for d in parent.iterdir():
        # Here d could also be a file
        if re.search(fr'{start}\d{{4}}', d.name):
            seq = int(d.name[1:])
            if seq >= next_run_index:
                next_run_index = seq + 1

    next_output_dir = parent.joinpath(f'{start}{next_run_index:04d}')
    next_output_dir.mkdir(parents=True, exist_ok=False)
    return next_output_dir


class WriterBase(abc.ABC):
    """Base class for HDF5 writer."""

    _FILE_ROOT_NAME = Template("$group-$seq.hdf5")

    # cap the number of sequence files
    _MAX_SEQUENCE = 99

    _IMAGE_CHUNK = (16, 512)

    def __init__(self, path: Union[str, Path], schema: dict, *,
                 chmod: bool = True,
                 group: int = 1,
                 chunk_size: int = 50,
                 max_events_per_file: int = 1000):
        """Initialization.

        :param path: Path of the simulation/run folder.
        :param schema: Data schema.
        :param chmod: True for changing the permission to 400 after
            finishing writing.
        :param group: Writer group (1-99).
        :param chunk_size: Size of the first dimension of a chunk in
            a dataset.
        :param max_events_per_file: Maximum events stored in a single file.

        :raise OSError if the file already exists.
        """
        self._path = path if isinstance(path, Path) else Path(path)
        self._fp = None
        self._chmod = chmod
        self._group = group
        if not isinstance(group, int) or group < 1 or group > 99:
            raise ValueError("group must be an integer within [1, 99]")
        self._chunk_size = chunk_size

        if max_events_per_file < chunk_size:
            raise ValueError(
                "max_events_per_file cannot be smaller than chunk_size")
        self._max_events_per_file = max_events_per_file

        self._schema = schema

        self._data_channels = []
        self._ids = []

        self._index = 0
        self._file_count = 0

    def _next_file(self) -> Path:
        return self._path.joinpath(self._FILE_ROOT_NAME.substitute(
            run=self._path.name.upper(),
            group=f"{self._group:02d}",
            seq=f"{self._file_count:06d}"))

    @abc.abstractmethod
    def _init_channel_data(self, channel: str) -> None:
        pass

    def _initialize_next_file(self) -> h5py.File:
        """Create a new file in the run folder."""
        self._fp = h5py.File(self._next_file(), 'w-')

        self._file_count += 1
        self._ids.clear()

        self._init_meta_data()
        for ch in self._data_channels:
            self._init_channel_data(ch)

        return self._fp

    def _init_meta_data(self):
        fp = self._fp
        fp.create_dataset("METADATA/createDate",
                          data=datetime.now().isoformat())
        fp.create_dataset("METADATA/updateDate",
                          data=datetime.now().isoformat())

    @abc.abstractmethod
    def write(self, id_: int, data: dict) -> None:
        """Write data into HDF5 file."""

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


class SimWriter(WriterBase):
    """Write simulated data in HDF5 files."""

    _FILE_ROOT_NAME = Template("SIM-$run-G$group-S$seq.hdf5")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: consider moving it into schema
        self._data_channels = ["control", "phasespace"]

    def _init_channel_data(self, channel: str) -> None:
        fp = self._fp
        schema = self._schema[channel]

        meta_ch = f"METADATA/{channel}Channel"
        fp.create_dataset(meta_ch,
                          dtype=h5py.string_dtype(),
                          shape=(len(schema),))

        for i, (k, v) in enumerate(schema.items()):
            fp[meta_ch][i] = k
            dtype = v['type']
            if dtype == 'phasespace':
                for col in Phasespace.columns:
                    fp.create_dataset(
                        f"{channel.upper()}/{col.upper()}/{k}",
                        shape=(self._chunk_size, v['macroparticles']),
                        dtype='<f8',
                        chunks=(self._chunk_size, v['macroparticles']),
                        maxshape=(self._max_events_per_file,
                                  v['macroparticles']))
            else:
                fp.create_dataset(
                    f"{channel.upper()}/{k}",
                    shape=(self._chunk_size,),
                    dtype=v['type'],
                    chunks=(self._chunk_size,),
                    maxshape=(self._max_events_per_file,))

    def write(self, id_: int, data: dict) -> None:
        """Write data from one simulation into the file.

        :param id_: Simulation ID.
        :param data: Data to be written.
        """
        fp = self._fp
        chunk_size = self._chunk_size
        idx = self._index

        if idx % self._max_events_per_file == 0:
            # close the current file
            self.close()
            # create a new one
            fp = self._initialize_next_file()
            self._index = idx = 0
        elif idx % chunk_size == 0:
            n_chunks = idx // chunk_size + 1
            new_size = min(n_chunks * chunk_size, self._max_events_per_file)
            for k in self._schema["control"]:
                fp[f"CONTROL/{k}"].resize(new_size, axis=0)
            for k in self._schema["phasespace"]:
                for col in Phasespace.columns:
                    fp[f"PHASESPACE/{col.upper()}/{k}"].resize(new_size, axis=0)

        for k, v in data['control'].items():
            fp[f"CONTROL/{k}"][idx] = v

        for k, v in data['phasespace'].items():
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
        self._ids.append(id_)
        self._index += 1

    def _finalize(self):
        """Override."""
        fp = self._fp
        fp.create_dataset("INDEX/simId", data=self._ids, dtype='u8')
        fp["METADATA/updateDate"][()] = datetime.now().isoformat()


class ExpWriter(WriterBase):
    """Write experimental data in HDF5 files."""

    _FILE_ROOT_NAME = Template("RAW-$run-G$group-S$seq.hdf5")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._data_channels = ["control", "diagnostic"]

    def _init_channel_data(self, channel: str) -> None:
        """Override."""
        fp = self._fp
        schema = self._schema[channel]

        meta_ch = f"METADATA/{channel}Channel"
        fp.create_dataset(meta_ch,
                          dtype=h5py.string_dtype(),
                          shape=(len(schema),))

        for i, (k, v) in enumerate(schema.items()):
            fp[meta_ch][i] = k
            dtype = v['type']
            # TODO: optimize chunk size
            if dtype == 'NDArray':
                shape = v['shape']
                if len(shape) > 2:
                    if shape[-1] > self._IMAGE_CHUNK[1] and \
                            shape[-2] > self._IMAGE_CHUNK[0]:
                        chunk_size = (1,) * (len(shape) - 1) + self._IMAGE_CHUNK
                    else:
                        chunk_size = True  # auto chunk
                else:
                    chunk_size = (self._chunk_size,) + shape
                fp.create_dataset(
                    f"{channel.upper()}/{k}",
                    shape=(self._chunk_size, *v['shape']),
                    dtype=v['dtype'],
                    chunks=chunk_size,
                    maxshape=(self._max_events_per_file, *v['shape']))
            else:
                fp.create_dataset(
                    f"{channel.upper()}/{k}",
                    shape=(self._chunk_size,),
                    dtype=v['type'],
                    chunks=(self._chunk_size,),
                    maxshape=(self._max_events_per_file,))

    def write(self, id_: int, data: dict) -> None:
        """Write correlated data from one macro-pulse into the file.

        :param id_: Macro-pulse ID.
        :param data: Data to be written.
        """
        fp = self._fp
        chunk_size = self._chunk_size
        idx = self._index

        if idx % self._max_events_per_file == 0:
            # close the current file
            self.close()
            # create a new one
            fp = self._initialize_next_file()
            self._index = idx = 0
        elif idx % chunk_size == 0:
            n_chunks = idx // chunk_size + 1
            new_size = min(n_chunks * chunk_size, self._max_events_per_file)
            for k in self._schema["control"]:
                fp[f"CONTROL/{k}"].resize(new_size, axis=0)
            for k in self._schema["diagnostic"]:
                fp[f"DIAGNOSTIC/{k}"].resize(new_size, axis=0)

        for k, v in data['control'].items():
            fp[f"CONTROL/{k}"][idx] = v
        for k, v in data['diagnostic'].items():
            fp[f"DIAGNOSTIC/{k}"][idx] = v

        self._ids.append(id_)
        self._index += 1

    def _finalize(self):
        """Override."""
        fp = self._fp
        fp.create_dataset("INDEX/pulseId", data=self._ids, dtype='u8')
        fp["METADATA/updateDate"][()] = datetime.now().isoformat()
