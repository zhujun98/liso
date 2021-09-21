"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
from pathlib import Path
import shutil
from typing import Union
import weakref


class TempSimulationDirectory:
    """Create temporary directories to hold simulation files."""

    def __init__(self, name: Union[str, Path]) -> None:
        """Initialization.

        :param name: full path of the temporary directory.

        :raises FileExistsError: If the directory already exists.
        """
        name = Path(name)

        # owner can read, write and execute
        # Raise FileExistsError if the directory already exists and
        # 'self._cleanup' will not be registered.
        name.mkdir(mode=0o700)

        self._name = name
        self._finalizer = weakref.finalize(self, self._cleanup, name)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self._name}>"

    def __enter__(self) -> Path:
        return self._name

    def __exit__(self, exc, value, tb):
        self.cleanup()

    @classmethod
    def _cleanup(cls, name: Path) -> None:
        shutil.rmtree(name)

    def cleanup(self) -> None:
        if self._finalizer.detach() is not None:
            shutil.rmtree(self._name)
