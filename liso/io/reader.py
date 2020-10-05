"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import abc

import pandas as pd

from .file_access import SimFileAccess, ExpFileAccess
from ..data_processing import Phasespace


class _DataCollectionBase:
    """A collection of simulated or experimental data."""
    def __init__(self, files):
        """Initialization

        :param list files: a list of FileAccess instances.
        """
        self._files = list(files)

    @abc.abstractmethod
    def info(self):
        """Print out the information of this data collection."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_path(cls, path):
        raise NotImplementedError

    @abc.abstractmethod
    def get_controls(self):
        """Return a pandas.DataFrame containing control data."""
        raise NotImplementedError


class SimDataCollection(_DataCollectionBase):
    """A collection of simulated data."""
    _FileAccess = SimFileAccess

    def __init__(self, files):
        super().__init__(files)

        self.control_sources = set()
        self.phasespace_sources = set()
        for fa in self._files:
            self.control_sources.update(fa.control_sources)
            self.phasespace_sources.update(fa.phasespace_sources)

        self.control_sources = frozenset(self.control_sources)
        self.phasespace_sources = frozenset(self.phasespace_sources)

        # this returns a list!
        self.sim_ids = sorted(set().union(*(f.sim_ids for f in files)))

    def info(self):
        """Override."""
        print('# of simulations:     ', len(self.sim_ids))

        print(f"\nControl sources ({len(self.control_sources)}):")
        for src in sorted(self.control_sources):
            print('  - ', src)

        print(f"\nPhasespace sources ({len(self.phasespace_sources)}):")
        for src in sorted(self.phasespace_sources):
            print('  - ', src)

    @classmethod
    def from_path(cls, path):
        files = [SimFileAccess(path)]
        return cls(files)

    def get_controls(self):
        """Override."""
        data = []
        for fa in self._files:
            df = pd.DataFrame.from_dict({
                k: v[()] for k, v in fa.file["CONTROL"].items()
            })
            df.set_index(fa.sim_ids, inplace=True)
            data.append(df)
        return pd.concat(data)

    def __getitem__(self, item):
        fa = self._find_data(item)
        ret = dict()
        index = item - 1
        for src in self.control_sources:
            ret[src] = fa.file["CONTROL"][src][index]
        for src in self.phasespace_sources:
            ret[src] = Phasespace.from_dict(
                {col.lower(): fa.file["PHASESPACE"][col][src][index]
                 for col in fa.file["PHASESPACE"]}
            )

        return ret

    def __iter__(self):
        for sid in self.sim_ids:
            yield sid, self.__getitem__(sid)

    def _find_data(self, item) -> SimFileAccess:
        for fa in self._files:
            if item in fa.sim_ids:
                return fa
        raise IndexError


def open_sim(filepath):
    return SimDataCollection.from_path(filepath)


class ExpDataCollection(_DataCollectionBase):
    """A collection of experimental data."""
    _FileAccess = SimFileAccess

    def __init__(self, files):
        super().__init__(files)

        self.control_sources = set()
        self.detector_sources = set()
        for fa in self._files:
            self.control_sources.update(fa.control_sources)
            self.detector_sources.update(fa.detector_sources)

        self.control_sources = frozenset(self.control_sources)
        self.detector_sources = frozenset(self.detector_sources)

        # this returns a list!
        self.pulse_ids = sorted(set().union(*(f.pulse_ids for f in files)))

    def info(self):
        """Override."""
        print('# of macro pulses:     ', len(self.pulse_ids))

        print(f"\nControl sources ({len(self.control_sources)}):")
        for src in sorted(self.control_sources):
            print('  - ', src)

        print(f"\ndetector sources ({len(self.detector_sources)}):")
        for src in sorted(self.detector_sources):
            print('  - ', src)

    @classmethod
    def from_path(cls, path):
        files = [ExpFileAccess(path)]
        return cls(files)

    def get_controls(self):
        """Override."""
        data = []
        for fa in self._files:
            df = pd.DataFrame.from_dict({
                k: v[()] for k, v in fa.file["CONTROL"].items()
            })
            df.set_index(fa.pulse_ids, inplace=True)
            data.append(df)
        return pd.concat(data)

    def __getitem__(self, item):
        fa = self._find_data(item)
        ret = dict()
        index = item - 1
        for src in self.control_sources:
            ret[src] = fa.file["CONTROL"][src][index]
        for src in self.detector_sources:
            ret[src] = Phasespace.from_dict(
                {col.lower(): fa.file["PHASESPACE"][col][src][index]
                 for col in fa.file["PHASESPACE"]}
            )

        return ret

    def __iter__(self):
        for pid in self.pulse_ids:
            yield pid, self.__getitem__(pid)

    def _find_data(self, item) -> ExpFileAccess:
        for fa in self._files:
            if item in fa.pulse_ids:
                return fa
        raise IndexError


def open_run(filepath):
    return ExpDataCollection.from_path(filepath)
