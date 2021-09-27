"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import abc
from typing import Dict, List


class MachineInterface(abc.ABC):
    """Base class for different machine interfaces."""

    @property
    @abc.abstractmethod
    def schema(self) -> dict:
        pass

    @abc.abstractmethod
    def write(self, *args, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def read(self, *args, **kwargs) -> List[tuple]:
        pass

    @abc.abstractmethod
    def query(self, *args, **kwargs) -> Dict[str, dict]:
        pass

    @abc.abstractmethod
    def safe_write(self, *args, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def parse_readout(self, *args, **kwargs) -> dict:
        pass
