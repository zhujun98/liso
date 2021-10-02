"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import abc
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, AsyncIterable, Dict, Iterable, List, Optional


class MachineInterface(abc.ABC):
    """Base class for different machine interfaces."""

    @property
    @abc.abstractmethod
    def schema(self) -> dict:
        pass

    @abc.abstractmethod
    async def awrite(self,
                     mapping: Dict[str, Any], *,
                     executor: Optional[ThreadPoolExecutor] = None) -> None:
        """Asynchronously write new value(s)."""

    @abc.abstractmethod
    async def aread(self, n: Optional[int] = None, *,
                    executor: Optional[ThreadPoolExecutor] = None) -> \
            AsyncIterable:
        """Asynchronously read new pulse/train value(s)."""

    @abc.abstractmethod
    def write(self, mapping: Dict[str, Any], *,
              executor: Optional[ThreadPoolExecutor] = None) -> None:
        """Write new value(s)."""

    @abc.abstractmethod
    def read(self, n: int, *, executor: Optional[ThreadPoolExecutor] = None)\
            -> List[tuple]:
        """Read new pulse/train values(s)"""

    @abc.abstractmethod
    def query(self, addresses: Optional[Iterable[str]] = None, *,
              executor: Optional[ThreadPoolExecutor] = None) -> Dict[str, dict]:
        """Read new value(s) which can be from different pulses/trains."""

    @abc.abstractmethod
    def parse_readout(self, readout: dict, *,
                      verbose: bool = True,
                      validate: bool = False) -> Dict[str, Any]:
        """Parse the readout for writing data to files."""

    @abc.abstractmethod
    @asynccontextmanager
    async def safe_awrite(self, addresses: Iterable[str], *,
                          executor: ThreadPoolExecutor) -> None:
        """Contextmanger for safely modifying machine setups."""
