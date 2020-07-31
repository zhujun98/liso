"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import os.path as osp

import h5py

from .file_access import FileAccess


class DataCollection:
    """A collection of

    """
    def __init__(self, files, data_ids=None):
        """Initialization

        :param list files:
        :param iterable data_ids: ids of the data set.
        """
        self._files = list(files)

        if data_ids is None:
            data_ids = sorted(set().union(*(f.data_ids for f in files)))
        self._data_ids = data_ids

    @staticmethod
    def _open_file(path):
        try:
            fa = FileAccess(path)
        except Exception as e:
            return osp.basename(path), str(e)
        else:
            return osp.basename(path), fa

    @classmethod
    def from_path(cls, path):
        files = [FileAccess(path)]
        return cls(files)
