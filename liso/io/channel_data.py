"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import numpy as np
from typing import Optional, Union

from .file_access import _FileAccessBase


class _AbstractPulseTrainData:
    def __init__(self):
        self._ids = []
        self._files = []

    def __getitem__(self, item):
        raise NotImplementedError

    def __iter__(self):
        for id_ in self._ids:
            yield self.__getitem__(id_)

    def from_id(self, id_: int):
        """Return pulse-train data from the specified pulse ID.

        :param id_: pulse ID.
        """
        return self.__getitem__(id_)

    def from_index(self, index: int):
        """Return pulse-train data from the nth pulse given index.

        :param index: pulse index.
        """
        return self.__getitem__(self._ids[index])

    def _find_data(self, id_) -> (_FileAccessBase, int):
        for fa in self._files:
            idx = (fa._ids == id_).nonzero()[0]
            if idx.size > 0:
                return fa, idx[0]
        raise KeyError


class ChannelData(_AbstractPulseTrainData):
    """Data for one single channel.

    This class should not be created directly.
    """
    def __init__(self, address: str, *, files: list, category: str, ids: list,
                 columns: Optional[Union[str, list]]):
        """Initialization.

        :param address: The address of the channel. For experimental data,
            this is the DOOCS channel name.
        :param files: A list of _FileAccessBase objects.
        :param category: Category of the data, which can be CONTROL,
            PHASESPACE (simulation) or DIAGNOSTIC (experiment).
        :param ids: Simulation/pulse IDs of the data.
        :param columns: Columns for the phasespace data. If None, all the
            columns are taken.
        """
        super().__init__()

        self._address = address
        self._files = files
        self._category = category

        if category == 'PHASESPACE':
            avail_columns = list(files[0].file['PHASESPACE'].keys())
            if columns is None:
                columns = avail_columns
            else:
                if isinstance(columns, str):
                    columns = [columns.upper()]
                else:
                    columns = [col.upper() for col in columns]

                for col in columns:
                    if col not in avail_columns:
                        raise ValueError(
                            f"{col} is not a valid phasespace column!")

            self._columns = columns

            # for consistency, here we only take the first phasespace column
            self._full_path = f"{category}/{columns[0]}/{address}"
            ds0 = files[0].file[self._full_path]
            self._entry_shape = (len(columns),) + ds0.shape[1:]
        else:
            self._columns = None  # for consistency

            self._full_path = f"{category}/{address}"
            ds0 = files[0].file[self._full_path]
            self._entry_shape = ds0.shape[1:]

        self._dtype = ds0.dtype

        self._ids = ids

    def __getitem__(self, id_):
        """Override."""
        fa, idx = self._find_data(id_)
        if self._category == 'PHASESPACE':
            category = self._category
            address = self._address
            out = np.empty(shape=self._entry_shape, dtype=self._dtype)
            for i, col in enumerate(self._columns):
                out[i] = fa.file[f"{category}/{col}/{address}"][idx]
            return out

        return fa.file[self._full_path][idx]

    def numpy(self):
        """Return data as a numpy array."""
        # TODO: improve the performance
        out = np.empty(shape=(len(self._ids), *self._entry_shape),
                       dtype=self._dtype)
        for i, item in enumerate(self):
            out[i] = item
        return out
