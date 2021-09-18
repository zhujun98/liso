"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import abc


class MachineInterface(abc.ABC):
    """Base class for different machine interfaces."""

    @property
    @abc.abstractmethod
    def schema(self):
        pass

    @abc.abstractmethod
    def write(self, *args, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def read(self, *args, **kwargs) -> tuple:
        pass
