"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import abc
from collections import defaultdict
import os
import os.path as osp

import numpy as np
import pandas as pd

from .channel_data import ChannelData
from .file_access import SimFileAccess, ExpFileAccess
from ..proc import Phasespace


class _DataCollectionBase:
    """A collection of simulated or experimental data."""

    _FileAccess = None

    def __init__(self, files):
        """Initialization

        :param list files: a list of FileAccess instances.
        """
        self._files = list(files)
        self._ids = []

        # channels are not ubiquitous in each file
        self._channel_files = defaultdict(list)

    @abc.abstractmethod
    def info(self):
        """Print out the information of this data collection."""
        raise NotImplementedError

    @classmethod
    def from_path(cls, path):
        """Construct a data collection from a single file.

        :param str path: file path.
        """
        files = [cls._FileAccess(path)]
        return cls(files)

    @classmethod
    def _open_file(cls, path):
        try:
            fa = cls._FileAccess(path)
        except Exception as e:
            return osp.basename(path), str(e)
        else:
            return osp.basename(path), fa

    @classmethod
    def from_paths(cls, paths):
        """Construct a data collection from a list of files.

        :param list paths: a list of file paths.
        """
        files = []
        for path in paths:
            fname, ret = cls._open_file(path)
            if isinstance(ret, cls._FileAccess):
                files.append(ret)
            else:
                print(f"Skipping path {fname}: {ret}")

        return cls(files)

    def get_controls(self, *, sorted=False):
        """Return control data in a Pandas.DataFrame.

        :param bool sorted: sort the index, which is indeed the ID, of the
            returned dataframe. This is sometime needed because the simulation
            data are not stored with the simulation IDs monotonically
            increasing.
        """
        data = []
        for fa in self._files:
            if 'METADATA/controlChannels' in fa.file:
                # backward compatibility
                control_channel_path = 'METADATA/controlChannels'
            else:
                control_channel_path = 'METADATA/controlChannel'

            ids = fa._ids
            df = pd.DataFrame.from_dict({
                ch: fa.file[f"CONTROL/{ch}"][:len(ids)]
                for ch in fa.file[control_channel_path]
            })
            df.set_index(ids, inplace=True)
            data.append(df)

        if sorted:
            return pd.concat(data).sort_index()
        return pd.concat(data)

    @abc.abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    def __iter__(self):
        for id_ in self._ids:
            yield self.__getitem__(id_)

    def from_index(self, index):
        """Return the data from the nth pulse given index.

        :param int index: pulse index.
        """
        return self.__getitem__(self._ids[index])

    def _find_data(self, id_) -> (ExpFileAccess, int):
        for fa in self._files:
            idx = (fa._ids == id_).nonzero()[0]
            if idx.size > 0:
                return fa, idx[0]
        raise KeyError

    @abc.abstractmethod
    def _get_channel_category(self, ch):
        raise NotImplementedError

    def channel(self, address, columns=None):
        """Return an array for a particular data field.

        :param str address: address of the channel.
        :param None/str/array-like columns: columns for the phasespace data.
            If None, all the columns are taken.
        """
        files = self._channel_files[address]
        if not files:
            raise KeyError(f"No data was found for channel: {address}")
        category = self._get_channel_category(address)
        return ChannelData(address,
                           files=files,
                           category=category,
                           ids=self._ids,
                           columns=columns)


class SimDataCollection(_DataCollectionBase):
    """A collection of simulated data."""
    _FileAccess = SimFileAccess

    def __init__(self, files):
        super().__init__(files)

        self.control_channels = set()
        self.phasespace_channels = set()
        for fa in self._files:
            self.control_channels.update(fa.control_channels)
            self.phasespace_channels.update(fa.phasespace_channels)
            for ch in (fa.control_channels | fa.phasespace_channels):
                self._channel_files[ch].append(fa)

        self.control_channels = frozenset(self.control_channels)
        self.phasespace_channels = frozenset(self.phasespace_channels)

        self.sim_ids = np.concatenate([f.sim_ids for f in files])
        self._ids = self.sim_ids

    def info(self):
        """Override."""
        print('# of simulations:     ', len(self.sim_ids))

        print(f"\nControl channels ({len(self.control_channels)}):")
        for src in sorted(self.control_channels):
            print('  - ', src)

        print(f"\nPhasespace channels ({len(self.phasespace_channels)}):")
        for src in sorted(self.phasespace_channels):
            print('  - ', src)

    def __getitem__(self, sim_id):
        fa, idx = self._find_data(sim_id)
        ret = dict()
        for ch in self.control_channels:
            ret[ch] = fa.file[f"CONTROL/{ch}"][idx]
        for ch in self.phasespace_channels:
            ret[ch] = Phasespace.from_dict(
                {col.lower(): fa.file[f"PHASESPACE/{col}/{ch}"][idx]
                 for col in fa.file["PHASESPACE"]}
            )

        return sim_id, ret

    def _get_channel_category(self, ch):
        return 'CONTROL' if ch in self.control_channels else 'PHASESPACE'


def open_sim(path):
    """Open simulation data from a single file or a directory.

    :param str path: file or directory path.
    """
    if osp.isfile(path):
        return SimDataCollection.from_path(path)

    paths = [osp.join(path, f) for f in os.listdir(path) if f.endswith('.hdf5')]
    if not paths:
        raise Exception(f"No HDF5 files found in {path}!")
    return SimDataCollection.from_paths(sorted(paths))


class ExpDataCollection(_DataCollectionBase):
    """A collection of experimental data."""
    _FileAccess = ExpFileAccess

    def __init__(self, files):
        super().__init__(files)

        self.control_channels = set()
        self.diagnostic_channels = set()
        for fa in self._files:
            self.control_channels.update(fa.control_channels)
            self.diagnostic_channels.update(fa.diagnostic_channels)
            for ch in (fa.control_channels | fa.diagnostic_channels):
                self._channel_files[ch].append(fa)

        self.control_channels = frozenset(self.control_channels)
        self.diagnostic_channels = frozenset(self.diagnostic_channels)

        self.pulse_ids = np.concatenate([f.pulse_ids for f in files])
        self._ids = self.pulse_ids

    def info(self):
        """Override."""
        print('# of macro pulses:     ', len(self.pulse_ids))

        print(f"\nControl channels ({len(self.control_channels)}):")
        for ch in sorted(self.control_channels):
            print('  - ', ch)

        print(f"\nInstrument channels ({len(self.diagnostic_channels)}):")
        for ch in sorted(self.diagnostic_channels):
            print('  - ', ch)

    def __getitem__(self, pulse_id):
        fa, idx = self._find_data(pulse_id)
        ret = dict()
        for ch in self.control_channels:
            ret[ch] = fa.file[f"CONTROL/{ch}"][idx]
        for ch in self.diagnostic_channels:
            ret[ch] = fa.file[f"DIAGNOSTIC/{ch}"][idx]

        return pulse_id, ret

    def _get_channel_category(self, ch):
        return 'CONTROL' if ch in self.control_channels else 'DIAGNOSTIC'


def open_run(path):
    """Open experimental data from a single file or a directory.

    :param str path: file or directory path.
    """
    if osp.isfile(path):
        return ExpDataCollection.from_path(path)

    paths = [osp.join(path, f) for f in os.listdir(path) if f.endswith('.hdf5')]
    if not paths:
        raise Exception(f"No HDF5 files found in {path}!")
    return ExpDataCollection.from_paths(sorted(paths))
