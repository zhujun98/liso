"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import time
from typing import Any, Optional, Type

from pydantic import ValidationError

try:
    from pydoocs import read as pydoocs_read
    from pydoocs import write as pydoocs_write
    from pydoocs import DoocsException, PyDoocsException
except ModuleNotFoundError:
    pydoocs_err = "pydoocs is required to communicate with a real " \
                  "machine using DOOCS control system!"

    def pydoocs_read(*args):
        raise ModuleNotFoundError(pydoocs_err)

    def pydoocs_write(*args):
        raise ModuleNotFoundError(pydoocs_err)

    class DoocsException(Exception):
        pass

    class PyDoocsException(Exception):
        pass

from ..exceptions import LisoRuntimeError
from ..logging import logger
from ..utils import profiler
from .machine_interface import MachineInterface
from .doocs_channels import DoocsChannel


class DoocsInterface(MachineInterface):
    """Interface for machines which uses DOOCS control system."""

    def __init__(self, facility_name: str, config: Optional[dict] = None):
        """Initialization.

        :param facility_name: Facility name.
        :param config: Config parameters for the facility.
        """
        super().__init__()

        self._facility_name = facility_name

        self._controls = OrderedDict()
        self._diagnostics = OrderedDict()
        self._controls_write = dict()
        self._non_event = set()

        self._last_correlated = 0

        self._timeout_read = 1.0
        self._timeout_write = 1.0

        if config is None:
            config = dict()

        tc = config.get("timeout.correlation")
        self._timeout_correlating = 2.0 if tc is None else tc

        irn = config.get("interval.read.non_event_data")
        self._interval_read_non_event_data = 1.0 if irn is None else irn

        irr = config.get("interval.read.retry")
        self._interval_read_retry = 0.1 if irr is None else irr

    @property
    def channels(self) -> list[str]:
        """Return a list of all DOOCS addresses."""
        return list(self._controls) + list(self._diagnostics)

    @property
    def controls(self) -> list[str]:
        """Return a list of DOOCS addresses for control data."""
        return list(self._controls)

    @property
    def diagnostics(self) -> list[str]:
        """Return a list of DOOCS addresses for diagnostic data."""
        return list(self._diagnostics)

    @property
    def schema(self) -> tuple[dict, dict]:
        """Return the schema of all DOOCS addresses."""
        return ({k: v.value_schema() for k, v in self._controls.items()},
                {k: v.value_schema() for k, v in self._diagnostics.items()})

    def _check_address(self, address: str) -> None:
        if address in self._controls:
            raise ValueError(f"{address} is an existing control channel!")

        if address in self._diagnostics:
            raise ValueError(f"{address} is an existing diagnostics channel!")

        if not address.startswith(self._facility_name):
            raise ValueError(f"{address} must start with {self._facility_name}")

    def add_control_channel(self, kls: Type[DoocsChannel],
                            read_address: str,
                            write_address: Optional[str] = None, *,
                            non_event: bool = False, **kwargs) -> None:
        """Add a DOOCS channel for control data.

        :param kls: A concrete DoocsChannel class.
        :param read_address: DOOCS read address.
        :param write_address: DOOCS write address. It will be set to the same
            as the read address if not given.
        :param non_event: True for a non-event-based channel (slow collector).
        :param kwargs: Keyword arguments which will be passed to the
            constructor of kls after address.

        Examples:
            >>> from liso import doocs_channels as dc
            >>> from liso import EuXFELInterface

            >>> m = EuXFELInterface()
            >>> m.add_control_channel(
            >>>     dc.FLOAT32,
            >>>     'XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/PHASE.SAMPLE',
            >>>     'XFEL.RF/LLRF.CONTROLLER/CTRL.GUN.I1/SP.PHASE')
        """
        self._check_address(read_address)
        self._controls[read_address] = kls(address=read_address, **kwargs)
        if write_address is None:
            write_address = read_address
        self._controls_write[read_address] = write_address
        if non_event:
            self._non_event.add(read_address)

    def add_diagnostic_channel(self, kls: Type[DoocsChannel], address: str, *,
                               non_event: bool = False, **kwargs) -> None:
        """Add a DOOCS channel to diagnostic data.

        :param kls: A concrete DoocsChannel class.
        :param address: DOOCS address.
        :param non_event: True for a non-event-based channel.
        :param kwargs: Keyword arguments which will be passed to the
            constructor of kls after address.

        Examples:
            >>> from liso import doocs_channels as dc
            >>> from liso import EuXFELInterface

            >>> m = EuXFELInterface()
            >>> m.add_diagnostic_channel(
            >>>     dc.IMAGE, 'XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ',
            >>>     shape=(1750, 2330), dtype='uint16')

        """
        self._check_address(address)
        self._diagnostics[address] = kls(address=address, **kwargs)
        if non_event:
            self._non_event.add(address)

    @staticmethod
    async def _cancel_all(futures):
        for fut in futures:
            fut.cancel()
            try:
                await fut
            except asyncio.CancelledError:
                pass

    async def _write_channel(self,
                             address: str,
                             value: Any,
                             loop: asyncio.AbstractEventLoop,
                             executor: ThreadPoolExecutor) -> bool:
        """Write a single channel and parse the result."""
        try:
            await loop.run_in_executor(
                executor, pydoocs_write, address, value)
            return True
        except ModuleNotFoundError as e:
            logger.error(repr(e))
            raise
        except (DoocsException, PyDoocsException) as e:
            logger.error(f"Failed to write {value} to {address}: {repr(e)}")
        except Exception as e:
            logger.error(f"Unexpected exception when writing {value} to "
                         f"{address}: {repr(e)}")
        return False

    async def _write(self,
                     mapping: dict[str, Any],
                     loop: asyncio.AbstractEventLoop,
                     executor: ThreadPoolExecutor) -> int:
        """Implementation of write."""
        tasks = [
            asyncio.create_task(self._write_channel(addr, v, loop, executor))
            for addr, v in mapping.items()
        ]

        failure_count = 0
        for fut in asyncio.as_completed(tasks, timeout=self._timeout_write):
            if not (await fut):
                failure_count += 1
        return failure_count

    @profiler("DOOCS interface write")
    def write(self, mapping: dict[str, Any], *,
              executor: Optional[ThreadPoolExecutor] = None,
              loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Write new value(s) to the given control channel(s).

        :param mapping: A mapping between DOOCS channel(s) and value(s).
        :param executor: ThreadPoolExecutor instance.
        :param loop: The event loop.

        :raises ModuleNotFoundError: If PyDOOCS cannot be imported.
        :raises LisoRuntimeError: If there is error when writing any channels.
        """
        if not mapping:
            return

        mapping_write = OrderedDict()
        for k, v in mapping.items():
            try:
                mapping_write[self._controls_write[k]] = v
            except KeyError:
                raise KeyError(f"Channel {k} is not found in the "
                               f"control channels.")
        # TODO: Validate the values to be written

        if executor is None:
            executor = ThreadPoolExecutor()
        if loop is None:
            loop = asyncio.get_event_loop()

        failure_count = loop.run_until_complete(
            self._write(mapping_write, loop, executor))
        if failure_count > 0:
            raise LisoRuntimeError(
                f"Failed to update {failure_count}/{len(mapping_write)} "
                f"channels ")

    def _validate_readout(self,
                          channels: dict[str, DoocsChannel],
                          readout: dict) -> dict[str, Any]:
        """Validate readout for given channels.
        
        :raises LisoRuntimeError
        """
        ret = dict()
        for address, ch in channels.items():
            ch_data = readout[address]
            ret[address] = ch_data
            if ch_data is not None:
                try:
                    ch.value = ch_data['data']  # validate
                except ValidationError as e:
                    raise LisoRuntimeError(repr(e))
        return ret

    async def _read_channel(self,
                            address: str,
                            loop: asyncio.AbstractEventLoop,
                            executor: ThreadPoolExecutor,
                            delay: float = 0) -> tuple(str, Any):
        """Read a single channel and parse the result.

        :raises ModuleNotFoundError: If PyDOOCS cannot be imported.
        """
        if delay > 0:
            await asyncio.sleep(delay)

        try:
            return (address,
                    await loop.run_in_executor(executor, pydoocs_read, address))
        except ModuleNotFoundError as e:
            logger.error(repr(e))
            raise
        except (DoocsException, PyDoocsException) as e:
            logger.warning(f"Failed to read data from {address}: {repr(e)}")
        except Exception as e:
            logger.error(f"Unexpected exception when reading from "
                         f"{address}: {repr(e)}")
        return address, None

    async def _read_correlated(self,
                               channels: list[str],
                               loop: asyncio.AbstractEventLoop,
                               executor: ThreadPoolExecutor) \
            -> tuple[int, dict[str, dict]]:
        """Read the first available correlated data from channels.

        :raises ModuleNotFoundError: If PyDOOCS cannot be imported.
        """
        n_events = len(self.channels) - len(self._non_event)
        cached = OrderedDict()

        tasks = dict()

        SENTINEL = object()
        if self._timeout_correlating is not None:
            tasks[asyncio.create_task(
                asyncio.sleep(self._timeout_correlating))] = SENTINEL

        tasks.update(
            {asyncio.create_task(
                self._read_channel(address, loop, executor)): address
                 for address in self.channels
             if address not in self._non_event}
        )

        correlated = dict()
        delay = self._interval_read_retry
        running = True
        while running:
            done, _ = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED)

            for fut in done:
                if tasks[fut] is SENTINEL:
                    running = False
                    continue

                address, ch_data = fut.result()
                if ch_data is not None:
                    pid = ch_data['macropulse']
                    if pid > self._last_correlated:
                        if pid not in cached:
                            cached[pid] = dict()
                        cached[pid][address] = ch_data

                        if len(cached[pid]) == n_events:
                            await self._cancel_all(tasks)

                            _, non_event_data = await self._read(
                                self._non_event, loop, executor)

                            for ne_addr, ne_data in non_event_data.items():
                                correlated[ne_addr] = ne_data

                            logger.info(
                                f"Correlated {len(self.channels)}"
                                f"({n_events}) channels with "
                                f"macropulse ID: {pid}")

                            self._last_correlated = pid
                            correlated.update(cached[pid])

                            return pid, correlated
                    elif pid == 0:
                        # FIXME: It is not 100% sure that data with
                        #        macropulse ID equal to 0 is from a
                        #        slow collector.
                        logger.warning(
                            f"Received data from channel {address} "
                            f"with macropulse == 0. It is recommended to "
                            f"add this channel as 'non_event'.")
                        delay = self._interval_read_non_event_data
                    elif pid < 0:
                        # TODO: document when a macropulse ID is -1
                        logger.warning(
                            f"Received data from channel {address} "
                            f"with illegal macropulse == {pid}.")
                    else:
                        logger.debug(
                            f"Received data from channel {address} "
                            f"with outdated macropulse ID: {pid}."
                        )

                del tasks[fut]
                tasks[asyncio.create_task(self._read_channel(
                    address, loop, executor, delay))] = address

        await self._cancel_all(tasks)
        return None, correlated

    async def _read(self,
                    channels: list[str],
                    loop: asyncio.AbstractEventLoop,
                    executor: ThreadPoolExecutor) \
            -> tuple[None, dict[key, dict]]:
        """Read data from channels without correlating them.

        :raises ModuleNotFoundError: If PyDOOCS cannot be imported.
        """
        tasks = [
            asyncio.create_task(self._read_channel(addr, loop, executor))
            for addr in channels
        ]

        rets = dict()
        for fut in asyncio.as_completed(tasks, timeout=self._timeout_read):
            address, data = await fut
            rets[address] = data
        return None, rets

    @profiler("DOOCS interface read")
    def read(self,
             executor: Optional[ThreadPoolExecutor] = None,
             loop: Optional[asyncio.AbstractEventLoop] = None,
             correlated: bool = True) -> dict:
        """Return readout value(s) of the diagnostics channel(s).

        :param executor: ThreadPoolExecutor instance.
        :param loop: The event loop.
        :param correlated: True for returning the latest group of data with
            the same train ID.

        :raises ModuleNotFoundError: If PyDOOCS cannot be imported.

        The returned data from each channel contains the following keys:
            data, macropulse, timestamp, type, miscellaneous
        """
        if executor is None:
            executor = ThreadPoolExecutor()
        if loop is None:
            loop = asyncio.get_event_loop()

        if correlated:
            pid, data =  loop.run_until_complete(
                self._read_correlated(self.channels, loop, executor))
            if pid is None:
                raise LisoRuntimeError("Failed to correlate all channel data!")
        else:
            pid, data =  loop.run_until_complete(
                self._read(self.channels, loop, executor))

        control_data = self._validate_readout(self._controls, data)
        diagnostic_data = self._validate_readout(self._diagnostics, data)

        return pid, control_data, diagnostic_data

    @staticmethod
    def _print_channel_data(title, data):
        print(f"{title}:\n" + "\n".join([f"- {k}: {v}" for k, v in data.items()]))

    def monitor(self, executor: Optional[ThreadPoolExecutor] = None) -> None:
        """Continuously monitoring the diagnostic channels.

        :param executor: ThreadPoolExecutor instance.
        """
        loop = asyncio.get_event_loop()
        try:
            while True:
                pid, controls, diagnostics = self.read(executor, loop=loop)

                print("-" * 80)
                print("Macropulse ID:", pid)
                self._print_channel_data("\nControl data", controls)
                self._print_channel_data("\nDiagnostics data", diagnostics)
                print("-" * 80)

                time.sleep(0.001)
        except KeyboardInterrupt:
            logger.info("Stopping monitoring ...")


class EuXFELInterface(DoocsInterface):
    def __init__(self, config: Optional[dict] = None):
        """Initialization.

        :param config: Config parameters for the facility.
        """
        super().__init__('XFEL', config)


class FLASHInterface(DoocsInterface):
    def __init__(self, config: Optional[dict] = None):
        """Initialization.

        :param config: Config parameters for the facility.
        """
        super().__init__('FLASH', config)
