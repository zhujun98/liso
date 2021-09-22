"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from pydantic import ValidationError

try:
    from pydoocs import read as pydoocs_read  # pylint: disable=import-error
    from pydoocs import write as pydoocs_write  # pylint: disable=import-error
    from pydoocs import DoocsException, PyDoocsException  # pylint: disable=import-error
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
from ..io import create_next_run_folder, ExpWriter
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
        self._pulse_interval = 0.1

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

        irr = config.get("interval.read.retry")
        self._interval_read_retry = self._pulse_interval if irr is None else irr

    @property
    def channels(self) -> List[str]:
        """Return a list of all DOOCS addresses."""
        return list(self._controls) + list(self._diagnostics)

    @property
    def controls(self) -> List[str]:
        """Return a list of DOOCS addresses for control data."""
        return list(self._controls)

    @property
    def diagnostics(self) -> List[str]:
        """Return a list of DOOCS addresses for diagnostic data."""
        return list(self._diagnostics)

    @property
    def schema(self) -> dict:
        """Return the schema of all DOOCS addresses."""
        return {
            "control": {
                k: v.value_schema() for k, v in self._controls.items()
            },
            "diagnostic": {
                k: v.value_schema() for k, v in self._diagnostics.items()
            }
        }

    def _check_address(self, address: str) -> None:
        if address in self._controls:
            raise ValueError(f"{address} is an existing control channel!")

        if address in self._diagnostics:
            raise ValueError(f"{address} is an existing diagnostics channel!")

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

    @staticmethod
    async def _write_channel(address: str,
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
            logger.error("Failed to write %s to %s: %s",
                         value, address, repr(e))
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Unexpected exception when writing %s to %s: %s",
                         value, address, repr(e))
        return False

    async def _write(self,
                     mapping: Dict[str, Any],
                     loop: asyncio.AbstractEventLoop,
                     executor: ThreadPoolExecutor) -> int:
        """Implementation of write."""
        tasks = [
            asyncio.create_task(self._write_channel(addr, v, loop, executor))
            for addr, v in mapping.items()
        ]

        failure_count = 0
        for fut in asyncio.as_completed(tasks, timeout=self._timeout_write):
            if not await fut:
                failure_count += 1
        return failure_count

    @profiler("DOOCS interface write")
    def write(self, mapping: Dict[str, Any], *,  # pylint: disable=arguments-differ
              loop: Optional[asyncio.AbstractEventLoop] = None,
              executor: Optional[ThreadPoolExecutor] = None) -> None:
        """Write new value(s) to the given control channel(s).

        :param mapping: A mapping between DOOCS channel(s) and value(s).
        :param loop: The event loop.
        :param executor: ThreadPoolExecutor instance.

        :raises ModuleNotFoundError: If PyDOOCS cannot be imported.
        :raises LisoRuntimeError: If there is error when writing any channels.
        """
        if not mapping:
            return

        # Validate should be done in DOOCS.
        try:
            mapping_write = {self._controls_write[k]: v
                             for k, v in mapping.items()}
        except KeyError as e:
            raise KeyError("Channel not found in the control channels") from e

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

    @staticmethod
    def _extract_readout(channels: Dict[str, DoocsChannel],
                         readout: dict) -> Dict[str, Any]:
        """Validate readout for given channels.

        :raises LisoRuntimeError: If validation fails.
        """
        ret = dict()
        for address, ch in channels.items():
            ch_data = readout[address]
            ret[address] = ch_data
            if ch_data is not None:
                try:
                    ch.value = ch_data['data']  # validate
                except ValidationError as e:
                    raise LisoRuntimeError("Validation error") from e
        return ret

    @staticmethod
    async def _read_channel(address: str,
                            loop: asyncio.AbstractEventLoop,
                            executor: ThreadPoolExecutor,
                            delay: float = 0) -> Tuple[str, Optional[dict]]:
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
            logger.error("Failed to read data from %s: {%s}", address, repr(e))
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Unexpected exception when reading from "
                         "{%s}: {%s}", address, repr(e))
        return address, None

    async def _read_correlated(self,  # pylint: disable=too-many-locals,too-many-branches
                               channels: List[str],
                               loop: asyncio.AbstractEventLoop,
                               executor: ThreadPoolExecutor) \
            -> Tuple[Optional[int], Dict[str, dict]]:
        """Read the first available correlated data from channels.

        :raises ModuleNotFoundError: If PyDOOCS cannot be imported.
        """
        tasks = dict()

        SENTINEL = object()
        if self._timeout_correlating is not None:
            tasks[asyncio.create_task(
                asyncio.sleep(self._timeout_correlating))] = SENTINEL

        tasks.update({
            asyncio.create_task(
                self._read_channel(address, loop, executor)): address
            for address in channels
        })

        n_channels = len(self.channels)
        n_nonevents = len(self._non_event)
        n_events = n_channels - n_nonevents

        ret = dict()
        cached = OrderedDict()
        latest_nonevent = dict()
        candidate_pids = set()
        correlated_pid = None
        running = True
        while running:  # pylint: disable=too-many-nested-blocks
            done, _ = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED)

            for fut in done:
                if tasks[fut] is SENTINEL:
                    running = False
                    break

                address, ch_data = fut.result()
                if ch_data is not None:
                    pid = ch_data['macropulse']
                    # Caveat: non-event data could have a normal macropulse ID.
                    if address in self._non_event or pid > self._last_correlated:
                        if address in self._non_event:
                            latest_nonevent[address] = ch_data
                        else:
                            if pid not in cached:
                                cached[pid] = dict()
                            cached[pid][address] = ch_data

                            if len(cached[pid]) == n_events:
                                candidate_pids.add(pid)

                        if candidate_pids and len(latest_nonevent) == n_nonevents:
                            correlated_pid = min(candidate_pids)
                            candidate_pids.remove(correlated_pid)

                            logger.info(
                                "Correlated %s (%s) channels with macropulse "
                                "ID: %s", n_channels, n_events, correlated_pid)

                            ret.update(cached[correlated_pid])
                            ret.update(latest_nonevent)
                            self._last_correlated = correlated_pid
                            running = False
                            break
                    elif pid == 0:
                        # FIXME: It is not 100% sure that data with
                        #        macropulse ID equal to 0 is from a
                        #        slow collector.
                        logger.warning(
                            "Received data from channel %s "
                            "with macropulse == 0. It is recommended to "
                            "add this channel as 'non_event'.", address)
                    elif pid < 0:
                        # TODO: document when a macropulse ID is -1
                        logger.warning(
                            "Received data from channel %s "
                            "with illegal macropulse == %s.", address, pid)
                    else:
                        logger.debug(
                            "Received data from channel %s "
                            "with outdated macropulse ID: %s.", address, pid
                        )

                del tasks[fut]
                tasks[asyncio.create_task(self._read_channel(
                    address, loop, executor,
                    delay=self._interval_read_retry))] = address

        await self._cancel_all(tasks)
        return correlated_pid, ret

    async def _read(self,
                    channels: List[str],
                    loop: asyncio.AbstractEventLoop,
                    executor: ThreadPoolExecutor) \
            -> Tuple[None, Dict[str, dict]]:
        """Read data from channels without correlating them.

        :raises ModuleNotFoundError: If PyDOOCS cannot be imported.
        """
        tasks = [
            asyncio.create_task(self._read_channel(address, loop, executor))
            for address in channels
        ]

        rets = dict()
        for fut in asyncio.as_completed(tasks, timeout=self._timeout_read):
            address, data = await fut
            rets[address] = data
        return None, rets

    @profiler("DOOCS interface read")
    def read(self,  # pylint: disable=arguments-differ
             loop: Optional[asyncio.AbstractEventLoop] = None,
             executor: Optional[ThreadPoolExecutor] = None,
             correlate: bool = True) -> Tuple[Optional[int], dict]:
        """Return readout value(s) of the diagnostics channel(s).

        :param loop: The event loop.
        :param executor: ThreadPoolExecutor instance.
        :param correlate: True for returning the latest group of data with
            the same train ID.

        :raises ModuleNotFoundError: If PyDOOCS cannot be imported.
        :raises LisoRuntimeError: If validation fails.

        The returned data from each channel contains the following keys:
            data, macropulse, timestamp, type, miscellaneous
        """
        if executor is None:
            executor = ThreadPoolExecutor()
        if loop is None:
            loop = asyncio.get_event_loop()

        if correlate:
            pid, data = loop.run_until_complete(
                self._read_correlated(self.channels, loop, executor))
            if pid is None:
                raise LisoRuntimeError("Failed to correlate all channel data!")
        else:
            pid, data = loop.run_until_complete(
                self._read(self.channels, loop, executor))

        control_data = self._extract_readout(self._controls, data)
        diagnostic_data = self._extract_readout(self._diagnostics, data)

        return pid, {
            "control": control_data,
            "diagnostic": diagnostic_data
        }

    @staticmethod
    def _print_channel_data(title: str, data: Dict[str, dict]) -> None:
        print(f"{title}:\n" + "\n".join([f"- {k}: {v}"
                                         for k, v in data.items()]))

    def acquire(self, output_dir: Union[str, Path] = "./", *,
                executor: Optional[ThreadPoolExecutor] = None,
                chmod: bool = True,
                group: int = 1):
        """Acquiring correlated data and saving it to HDF5 files.

        :param output_dir: Directory in which data for each run is
            stored in in its own sub-directory.
        :param executor: ThreadPoolExecutor instance.
        :param chmod: True for changing the permission to 400 after
            finishing writing.
        :param group: Writer group.
        """
        output_dir = create_next_run_folder(output_dir)

        logger.info("Starting acquiring data and saving data to %s",
                    output_dir.resolve())

        loop = asyncio.get_event_loop()
        with ExpWriter(output_dir,
                       schema=self.schema,
                       chmod=chmod,
                       group=group) as writer:
            try:
                while True:
                    pid, data = self.read(loop, executor)
                    ret = dict()
                    for key, item in data.items():
                        ret[key] = {k: v['data'] for k, v in item.items()}
                    writer.write(pid, ret)
            except KeyboardInterrupt:
                logger.info("Stopping data acquisition ...")

    def monitor(self,
                executor: Optional[ThreadPoolExecutor] = None,
                correlate: bool = False) -> None:
        """Continuously monitoring the diagnostic channels.

        :param executor: ThreadPoolExecutor instance.
        :param correlate: True for correlating all channel data.
        """
        loop = asyncio.get_event_loop()
        try:
            while True:
                # The readout must be validated and this is rational of
                # having the schema for each channel.
                pid, data = self.read(
                    loop, executor, correlate=correlate)

                # FIXME: change print to log?
                print("-" * 80)
                print("Macropulse ID:", pid)
                self._print_channel_data(
                    "\nControl data", data['control'])
                self._print_channel_data(
                    "\nDiagnostics data", data['diagnostic'])
                print("-" * 80)

                if correlate:
                    time.sleep(0.001)
                else:
                    time.sleep(1.0)
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
