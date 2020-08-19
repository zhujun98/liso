"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""

# This module is implemented based on:
# https://github.com/European-XFEL/EXtra-data/blob/master/extra_data/file_access.py

import os
import os.path as osp
import resource

from collections import defaultdict, OrderedDict
from weakref import WeakValueDictionary

import h5py

# Track all FileAccess objects - {path: FileAccess}
_file_access_registry = WeakValueDictionary()


class OpenFilesLimiter(object):

    def __init__(self, n_max=128):
        self._n_max = n_max

        self._cache = OrderedDict()

    @property
    def cache(self):
        return OrderedDict.fromkeys(
            path for path in self._cache if path in _file_access_registry
        )

    def n_opened(self):
        """Return the number of currently opened files."""
        return len(self.cache)

    def close_old_files(self):
        if len(self._cache) <= self._n_max:
            return

        # Now check how many paths still have an existing FileAccess object
        n = self.n_opened()
        while n > self._n_max:
            path, _ = self._cache.popitem(last=False)
            file_access = _file_access_registry.get(path, None)
            if file_access is not None:
                file_access.close()
            n -= 1

    def touch(self, filename):
        """
        Add/move the touched file to the end of the `cache`.
        If adding a new file takes it over the limit of open files, another file
        will be closed.

        For use of the file cache, FileAccess should use `touch(filename)` every time
        it provides the underying instance of `h5py.File` for reading.
        """
        if filename in self._cache:
            self._cache.move_to_end(filename)
        else:
            self._cache[filename] = None
            self.close_old_files()

    def closed(self, filename):
        """Discard a closed file from the cache"""
        self._cache.pop(filename, None)


def _init_open_files_limiter():
    # Raise the limit for open files (1024 -> 4096 on Maxwell)
    nofile = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (nofile[1], nofile[1]))
    maxfiles = nofile[1] // 2
    return OpenFilesLimiter(maxfiles)


_open_files_limiter = _init_open_files_limiter()


class FileAccess:
    """Access an HDF5 file.

    This does not necessarily keep the real file open, but opens it on demand.
    It assumes that the file is not changing on disk while this object exists.
    """
    _file = None

    def __new__(cls, filename):
        filename = osp.abspath(filename)
        instance = _file_access_registry.get(filename, None)
        if instance is None:
            instance = super().__new__(cls)
            _file_access_registry[filename] = instance
        return instance

    def __init__(self, filename):
        self.filename = osp.abspath(filename)

        tid_data = self.file['INDEX/trainId'][:]
        self.train_ids = tid_data[tid_data != 0]

        self.control_sources, self.instrument_sources = self._read_data_sources()

        # Store the stat of the file as it was when we read the metadata.
        # This is used by the run files map.
        self.metadata_fstat = os.stat(self.file.id.get_vfd_handle())

        # {(file, source, group): (firsts, counts)}
        self._index_cache = {}
        # {source: set(keys)}
        self._keys_cache = {}
        # {source: set(keys)} - including incomplete sets
        self._known_keys = defaultdict(set)

    @property
    def file(self):
        _open_files_limiter.touch(self.filename)
        if self._file is None:
            self._file = h5py.File(self.filename, 'r')
        return self._file

    def close(self):
        """Close the HDF5 file this refers to.
        The file may not actually be closed if there are still references to
        objects from it, e.g. while iterating over trains. This is what HDF5
        calls 'weak' closing.
        """
        if self._file:
            self._file = None
        _open_files_limiter.closed(self.filename)

    def __repr__(self):
        return "{}({})".format(type(self).__name__, repr(self.filename))
