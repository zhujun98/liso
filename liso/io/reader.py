"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import abc
from collections import defaultdict

import pandas as pd

from .channel_data import ChannelData
from .file_access import SimFileAccess, ExpFileAccess
from ..data_processing import Phasespace


class _DataCollectionBase:
    """A collection of simulated or experimental data."""
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
    @abc.abstractmethod
    def from_path(cls, path):
        raise NotImplementedError

    def get_controls(self):
        """Return control data in a Pandas.DataFrame."""
        data = []
        for fa in self._files:
            df = pd.DataFrame.from_dict({
                ch: fa.file[f"CONTROL/{ch}"][()]
                for ch in fa.file["METADATA/controlChannels"]
            })
            df.set_index(fa._ids, inplace=True)
            data.append(df)
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

        # this returns a list!
        self.sim_ids = sorted(set().union(*(f.sim_ids for f in files)))
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

    @classmethod
    def from_path(cls, path):
        files = [SimFileAccess(path)]
        return cls(files)

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


def open_sim(filepath):
    return SimDataCollection.from_path(filepath)


class ExpDataCollection(_DataCollectionBase):
    """A collection of experimental data."""
    _FileAccess = SimFileAccess

    def __init__(self, files):
        super().__init__(files)

        self.control_channels = set()
        self.detector_channels = set()
        for fa in self._files:
            self.control_channels.update(fa.control_channels)
            self.detector_channels.update(fa.detector_channels)
            for ch in (fa.control_channels | fa.detector_channels):
                self._channel_files[ch].append(fa)

        self.control_channels = frozenset(self.control_channels)
        self.detector_channels = frozenset(self.detector_channels)

        # this returns a list!
        self.pulse_ids = sorted(set().union(*(f.pulse_ids for f in files)))
        self._ids = self.pulse_ids

    def info(self):
        """Override."""
        print('# of macro pulses:     ', len(self.pulse_ids))

        print(f"\nControl channels ({len(self.control_channels)}):")
        for ch in sorted(self.control_channels):
            print('  - ', ch)

        print(f"\ndetector channels ({len(self.detector_channels)}):")
        for ch in sorted(self.detector_channels):
            print('  - ', ch)

    @classmethod
    def from_path(cls, path):
        files = [ExpFileAccess(path)]
        return cls(files)

    def __getitem__(self, pulse_id):
        fa, idx = self._find_data(pulse_id)
        ret = dict()
        for ch in self.control_channels:
            ret[ch] = fa.file[f"CONTROL/{ch}"][idx]
        for ch in self.detector_channels:
            ret[ch] = fa.file[f"DETECTOR/{ch}"][idx]

        return pulse_id, ret

    def _get_channel_category(self, ch):
        return 'CONTROL' if ch in self.control_channels else 'DETECTOR'


def open_run(filepath):
    return ExpDataCollection.from_path(filepath)
