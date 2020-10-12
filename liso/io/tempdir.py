"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import os
import shutil
import weakref


class TempSimulationDirectory(object):
    """Create temporary directories to hold simulation files."""

    def __init__(self, folder, *, delete_old=False):
        """Initialization.

        :param str folder: full path of the temporary directory.
        :param bool delete_old: True for deleting the folder if it already
            exists.
        """
        if delete_old and os.path.isdir(folder):
            shutil.rmtree(folder)

        # owner can read, write and execute
        # Raise FileExistsError if the directory already exists and
        # 'self._cleanup' will not be registered.
        os.mkdir(folder, mode=0o700)
        self._name = folder

        self._finalizer = weakref.finalize(self, self._cleanup, self._name)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self._name}>"

    def __enter__(self):
        return self._name

    def __exit__(self, exc, value, tb):
        self.cleanup()

    @classmethod
    def _cleanup(cls, name):
        shutil.rmtree(name)

    def cleanup(self):
        if self._finalizer.detach() is not None:
            shutil.rmtree(self._name)
