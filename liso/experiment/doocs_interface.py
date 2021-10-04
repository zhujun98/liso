"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
import random
import time
from typing import (
    Any, AsyncIterable, Awaitable, Callable, Dict, Iterable, List, Optional,
    FrozenSet, Tuple, Type, Union
)

from pydantic import ValidationError
from sortedcontainers import SortedDict

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
from .doocs_channels import AnyDoocsChannel, DoocsChannel


class Correlator:

    class Timer:
        def __init__(self, timeout: Optional[float], *, callback: Callable):
            self._timeout = timeout
            self._task = asyncio.create_task(self._countdown())
            self._callback = callback

        async def _countdown(self):
            if self._timeout is not None:
                await asyncio.sleep(self._timeout)
                self._callback()

        async def cancel(self):
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def __init__(self, *,
                 timeout: float,
                 retry_after: float,
                 event_buffer_size: int):
        self._timeout = timeout
        self._retry_after = retry_after
        self._event_buffer_size = event_buffer_size

        self._last_correlated = 0

        self._running = False

    @staticmethod
    async def _cancel_all(futures):
        for fut in futures:
            fut.cancel()
            try:
                await fut
            except asyncio.CancelledError:
                pass

    async def _collect_event(
            self,
            channels: set,
            buffer: SortedDict,
            ready: deque, *,
            read: Callable[..., Awaitable[Tuple[str, dict]]]) -> None:
        tasks = {
            asyncio.create_task(read(address)): address
            for address in channels
        }

        while self._running:  # pylint: disable=too-many-nested-blocks
            done, _ = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED)

            for fut in done:
                address, ch_data = fut.result()
                if ch_data is not None:
                    pid = ch_data['macropulse']
                    # Caveat: non-event data could have a normal macropulse ID.
                    if pid > self._last_correlated:
                        if pid not in buffer:
                            buffer[pid] = dict()
                        buffer[pid][address] = ch_data

                        # prevent the buffer from growing infinitely
                        if len(buffer) > self._event_buffer_size:
                            pid_removed, _ = buffer.popitem(index=0)
                            if ready and pid_removed == ready[0]:
                                ready.popleft()
                            logger.warning("Buffer full: drop macropulse %s",
                                           pid_removed)

                        # correlated
                        if pid in buffer and len(buffer[pid]) == len(channels):
                            self._last_correlated = pid
                            ready.append(pid)

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
                if self._running:
                    tasks[asyncio.create_task(read(
                        address, delay=self._retry_after))] = address

        await self._cancel_all(tasks)

    async def _collect_non_event(
            self,
            channels: set,
            buffer: dict,
            ready: deque, *,
            read: Callable[..., Awaitable[Tuple[str, dict]]]) -> None:

        if not channels:
            ready.append(object())
            return

        tasks = dict()

        tasks.update({
            asyncio.create_task(read(address)): address
            for address in channels
        })

        while self._running:
            done, _ = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED)

            for fut in done:
                address, ch_data = fut.result()
                if ch_data is not None:
                    buffer[address] = ch_data

                if not ready and len(buffer) == len(channels):
                    ready.append(object())

                del tasks[fut]
                tasks[asyncio.create_task(read(
                    address, delay=self._retry_after))] = address

        await self._cancel_all(tasks)

    def _stop(self):
        self._running = False

    async def correlate(self, n: Optional[int], event: set, non_event: set, *,
                        read: Callable[..., Awaitable[Tuple[str, dict]]],
                        executor: ThreadPoolExecutor):
        """Correlate all the given channels.

        :param n: Number of pulses (trains) to collect before returning. None
            for running in an infinite loop.
        :param event: Names of event channels.
        :param non_event: Names of non-event channels.
        :param read: Function for reading the channel data.
        :param executor: ThreadPoolExecutor instance.
        """
        event_buffer = SortedDict()
        event_ready = deque()
        non_event_buffer = dict()
        non_event_ready = deque()

        loop = asyncio.get_running_loop()
        read = partial(read, loop=loop, executor=executor)

        self._running = True

        event_task = asyncio.create_task(self._collect_event(
            event, event_buffer, event_ready, read=read))
        non_event_task = asyncio.create_task(self._collect_non_event(
            non_event, non_event_buffer, non_event_ready, read=read))

        timeout = None if n is None else self._timeout * n
        timer = self.Timer(timeout, callback=self._stop)

        count = 0
        while self._running:  # pylint: disable=too-many-nested-blocks
            if not non_event_ready or not event_ready:
                await asyncio.sleep(self._retry_after)
                continue

            while event_ready:
                pid = event_ready.popleft()
                while True:
                    key, data = event_buffer.popitem(index=0)
                    if key == pid:
                        break
                data.update(non_event_buffer)

                yield pid, data
                count += 1

                if n is not None and count == n:
                    self._running = False

        await timer.cancel()
        await event_task
        await non_event_task


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

        self._channels = dict()
        self._catelog = {
            "control": set(),
            "diagnostic": set()
        }
        self._event = set()
        self._non_event = set()

        self._control_write = dict()

        self._timeout_read = 1.0
        self._timeout_write = 1.0

        self._validation_prob = 0.1

        if config is None:
            config = dict()

        self._corr = Correlator(
            timeout=config.get("timeout.correlation", 2.0),
            retry_after=config.get("interval.read.retry", 0.01),
            event_buffer_size=config.get("buffer.event.size", 50)
        )

    @property
    def channels(self) -> FrozenSet[str]:
        """Return a set of all DOOCS addresses."""
        return frozenset(self._channels)

    @property
    def control_channels(self) -> FrozenSet[str]:
        """Return a set of DOOCS addresses for control channels."""
        return frozenset(self._catelog["control"])

    @property
    def diagnostic_channels(self) -> FrozenSet[str]:
        """Return a set of DOOCS addresses for diagnostic channels."""
        return frozenset(self._catelog["diagnostic"])

    @property
    def schema(self) -> dict:
        """Return the schema of all DOOCS addresses."""
        ret = dict()
        for cat, channels in self._catelog.items():
            ret[cat] = {
                k: v.value_schema() for k, v in self._channels.items()
                if k in channels
            }
        return ret

    def _check_address(self, address: str) -> None:
        if address in self._channels:
            raise ValueError(f"{address} is an existing channel!")

    def add_control_channel(self, address: str,
                            kls: Type[DoocsChannel] = AnyDoocsChannel, *,
                            write_address: Optional[str] = None,
                            non_event: bool = False, **kwargs) -> None:
        """Add a DOOCS channel for control data.

        :param address: DOOCS read address.
        :param kls: A concrete DoocsChannel class.
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
            >>>     'XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/PHASE.SAMPLE',
            >>>     dc.FLOAT32,
            >>>     write_address='XFEL.RF/LLRF.CONTROLLER/CTRL.GUN.I1/SP.PHASE')
        """
        self._check_address(address)
        self._channels[address] = kls(address=address, **kwargs)
        self._catelog["control"].add(address)
        if write_address is None:
            write_address = address
        self._control_write[address] = write_address
        if non_event:
            self._non_event.add(address)
        else:
            self._event.add(address)

    def add_diagnostic_channel(self, address: str,
                               kls: Type[DoocsChannel] = AnyDoocsChannel, *,
                               non_event: bool = False, **kwargs) -> None:
        """Add a DOOCS channel to diagnostic data.

        :param address: DOOCS address.
        :param kls: A concrete DoocsChannel class.
        :param non_event: True for a non-event-based channel.
        :param kwargs: Keyword arguments which will be passed to the
            constructor of kls after address.

        Examples:
            >>> from liso import doocs_channels as dc
            >>> from liso import EuXFELInterface

            >>> m = EuXFELInterface()
            >>> m.add_diagnostic_channel(
            >>>     'XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ', dc.ARRAY,
            >>>     shape=(1750, 2330), dtype='uint16')

        """
        self._check_address(address)
        self._channels[address] = kls(address=address, **kwargs)
        self._catelog["diagnostic"].add(address)
        if non_event:
            self._non_event.add(address)
        else:
            self._event.add(address)

    @staticmethod
    async def _write_channel(address: str,
                             value: Any, *,
                             loop: Optional[asyncio.AbstractEventLoop] = None,
                             executor: ThreadPoolExecutor) -> bool:
        """Write a single channel and parse the result."""
        if loop is None:
            loop = asyncio.get_running_loop()

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
        except asyncio.CancelledError:
            logger.debug("Task for writing %s to %s was cancelled",
                         value, address)
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Unexpected exception when writing %s to %s: %s",
                         value, address, repr(e))
        return False

    async def _write(self, mapping, *,
                     executor: Optional[ThreadPoolExecutor] = None) -> None:
        if not mapping:
            return

        loop = asyncio.get_running_loop()

        if executor is None:
            executor = ThreadPoolExecutor()

        tasks = [
            asyncio.create_task(self._write_channel(
                addr, v, loop=loop, executor=executor))
            for addr, v in mapping.items()
        ]

        failure_count = 0
        for fut in asyncio.as_completed(tasks, timeout=self._timeout_write):
            if not await fut:
                failure_count += 1

        if failure_count > 0:
            raise LisoRuntimeError(
                f"Failed to update {failure_count}/{len(mapping)} "
                f"channels ")

    async def awrite(self,
                     mapping: Dict[str, Any], *,
                     executor: Optional[ThreadPoolExecutor] = None) -> None:
        """Asynchronously write new value(s) to the given control channel(s).

        :param mapping: A mapping between DOOCS channel(s) and value(s). These
            channels must be among the existing control channels and the new
            value will actually be written into the corresponding DOOCS write
            address. See :meth:`~liso.experiment.DoocsInterface.add_control_channel`
        :param executor: ThreadPoolExecutor instance.

        :raises LisoRuntimeError: If there is error when writing any channels.
        """
        try:
            mapping_write = {self._control_write[k]: v
                             for k, v in mapping.items()}
        except KeyError as e:
            raise KeyError("Channel not found in the control channels") from e

        await self._write(mapping_write, executor=executor)

    @profiler("DOOCS interface write")
    def write(self, mapping: Dict[str, Any], *,
              executor: Optional[ThreadPoolExecutor] = None) -> None:
        """Write new value(s) to the given control channel(s).

        See :meth:`~liso.experiment.DoocsInterface.awrite`.
        """
        asyncio.run(self.awrite(mapping, executor=executor))

    def parse_readout(self, readout: dict, *,
                      verbose: bool = True,
                      validate: bool = False) -> Dict[str, Any]:
        """Parse readout.

        :param readout: Readout data.
        :param verbose: True for keeping the whole DOOCS message. Otherwise,
            only the 'data' of the message will be kept.
        :param validate: True for validate the readout 'data'.

        :raises TypeError: If validation fails.
        """
        ret = dict()
        for cat, channels in self._catelog.items():
            sub_ret = dict()
            for address in channels:
                ch_data = readout[address]
                ch_value = None if ch_data is None else ch_data['data']
                sub_ret[address] = ch_data if verbose else ch_value

                # Validation does not always happen due to
                # performance consideration.
                if validate and ch_value is not None \
                        and random.random() < self._validation_prob:
                    try:
                        self._channels[address].value = ch_value
                    except ValidationError:
                        raise TypeError(
                            f"Validation error of channel: {address}, "
                            f"expected: {self._channels[address].value_schema()}, "
                            f"actual: {ch_value}")
            ret[cat] = sub_ret
        return ret

    @staticmethod
    async def _read_channel(address: str, *,
                            delay: float = 0,
                            loop: Optional[asyncio.AbstractEventLoop] = None,
                            executor: ThreadPoolExecutor) -> Tuple[str, Optional[dict]]:
        """Read the data from a single channel.

        The returned data from each channel contains the following keys:
            data, macropulse, timestamp, type, miscellaneous
        """
        if delay > 0:
            await asyncio.sleep(delay)

        if loop is None:
            loop = asyncio.get_running_loop()

        try:
            data = await loop.run_in_executor(executor, pydoocs_read, address)
            return address, data

        except ModuleNotFoundError as e:
            logger.error(repr(e))
            raise
        except (DoocsException, PyDoocsException) as e:
            logger.error("Failed to read data from %s: %s", address, repr(e))
        except asyncio.CancelledError:
            logger.debug("Task for reading from %s was cancelled", address)
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Unexpected exception when reading from "
                         "%s: %s", address, repr(e))
        return address, None

    async def _query(self, addresses: Optional[Iterable[str]], *,
                     executor: Optional[ThreadPoolExecutor] = None) -> Dict[str, dict]:
        loop = asyncio.get_running_loop()
        if executor is None:
            executor = ThreadPoolExecutor()

        tasks = [
            asyncio.create_task(self._read_channel(
                address, loop=loop, executor=executor))
            for address in addresses
        ]

        ret = dict()
        for fut in asyncio.as_completed(tasks, timeout=self._timeout_read):
            address, data = await fut
            ret[address] = data
        return ret

    @profiler("DOOCS interface query")
    def query(self, addresses: Optional[Iterable[str]] = None, *,
              executor: Optional[ThreadPoolExecutor] = None) -> Dict[str, dict]:
        """Read data from channels once without correlating them.

        :param addresses: DOOCS addresses. None for reading all the existing
            channels.
        :param executor: ThreadPoolExecutor instance.
        """
        if addresses is None:
            addresses = self.channels

        return asyncio.run(self._query(addresses, executor=executor))

    async def aread(self, n: Optional[int] = None, *,
                    executor: Optional[ThreadPoolExecutor] = None) -> \
            AsyncIterable[Tuple[int, Dict[str, dict]]]:
        """Asynchronously read correlated data from all the channels.

        :param n: Number of correlated pulses to return. None for generating
            data infinitely.
        :param executor: ThreadPoolExecutor instance.

        :raises LisoRuntimeError: If there is no event channel.
        """
        if not self._event:
            raise LisoRuntimeError("At least one event channel is required to "
                                   "provide macropulse ID.")

        if executor is None:
            executor = ThreadPoolExecutor()

        async for data in self._corr.correlate(
                n, self._event, self._non_event,
                read=self._read_channel,
                executor=executor):
            yield data

    async def _read(self,
                    n: int,
                    executor: Optional[ThreadPoolExecutor] = None)\
            -> List[Tuple[int, dict]]:
        ret = []
        async for pid, data in self.aread(n, executor=executor):
            ret.append((pid, data))
        return ret

    @profiler("DOOCS interface read")
    def read(self, n: int, *, executor: Optional[ThreadPoolExecutor] = None)\
            -> List[Tuple[int, dict]]:
        """Read correlated data from all the channels.

        See :meth:`~liso.experiment.DoocsInterface.aread`.
        """
        return asyncio.run(self._read(n, executor))

    @asynccontextmanager
    async def safe_awrite(self, addresses: Iterable[str], *,  # pylint: disable=arguments-differ
                          executor: Optional[ThreadPoolExecutor] = None) -> None:
        """Context manager for safe asynchronous write.

        The initial value will be restored at exit.

        :raises KeyError: If any input address does not belong to the control
            channel.
        :raises LisoRuntimeError: If there is error when reading the initial
            values or restoring any channel at exit.
        """
        def _format_machine_setup(mapping):
            ret = ""
            for k_, v_ in mapping.items():
                ret += f"\n{k_}: {v_}"
            return ret

        try:
            write_addresses = [self._control_write[addr] for addr in addresses]
        except KeyError as e:
            raise KeyError("Channel not found in the control channels") from e

        if executor is None:
            executor = ThreadPoolExecutor()
        mapping_write = await self._query(write_addresses, executor=executor)

        for k, v in mapping_write.items():
            if v is None:
                raise LisoRuntimeError(f"Failed to read data from channel {k}")
            mapping_write[k] = v['data']

        logger.info("Initial machine setup: %s",
                    _format_machine_setup(mapping_write))

        try:
            yield
        finally:
            try:
                await self._write(mapping_write, executor=executor)
            except LisoRuntimeError as e:
                raise LisoRuntimeError(
                    "Failed to restore the machine setup") from e

            logger.info("Machine setup restored: %s",
                        _format_machine_setup(mapping_write))

    @staticmethod
    def _print_channel_data(title: str, data: Dict[str, dict]) -> None:
        print(f"{title}:\n" + "\n".join([f"- {k}: {v}"
                                         for k, v in data.items()]))

    async def _acquire_forever(self,
                               writer: ExpWriter,
                               executor: Optional[ThreadPoolExecutor] = None) -> None:
        count = 0
        try:
            async for pid, data in self.aread(executor=executor):
                data = self.parse_readout(data, verbose=False, validate=True)
                writer.write(pid, data)
                count += 1
                if count % 100 == 0:
                    logger.info("%s pulses acquired", count)
                time.sleep(0)  # for unittest
        finally:
            logger.info("Saved %s pulse(s) in total", count)

    def acquire(self, output_dir: Union[str, Path] = "./", *,
                executor: Optional[ThreadPoolExecutor] = None,
                chmod: bool = True,
                group: int = 1) -> None:
        """Acquiring correlated data and saving it to HDF5 files.

        :param output_dir: Directory in which data for each run is
            stored in in its own sub-directory.
        :param executor: ThreadPoolExecutor instance.
        :param chmod: True for changing the permission to 400 after
            finishing writing.
        :param group: Writer group.

        :raises LisoRuntimeError: If there is no event channel.
        """
        output_dir = create_next_run_folder(output_dir)

        logger.info("Starting acquiring data and saving data to %s",
                    output_dir.resolve())

        with ExpWriter(output_dir,
                       schema=self.schema,
                       chmod=chmod,
                       group=group) as writer:
            try:
                asyncio.run(self._acquire_forever(writer, executor))
            except KeyboardInterrupt:
                logger.info("Stopping data acquisition ...")

    def monitor(self, *,
                executor: Optional[ThreadPoolExecutor] = None,
                correlate: bool = False,
                verbose: bool = True,
                app: bool = False) -> None:
        """Continuously monitoring the diagnostic channels.

        :param executor: ThreadPoolExecutor instance.
        :param correlate: True for correlating all channel data.
        :param verbose: True for printing out the full DOOCS message.
            Otherwise, only 'data' is printed out.
        :param app: True for used in app with all channels belong to
            diagnostic.

        :raises LisoRuntimeError: If correlate is True and there is no
            event channel.
        """
        def _print_result(pid, data):
            data = self.parse_readout(data, verbose=verbose)

            print("-" * 80)
            print("Macropulse ID:", pid)
            for cat in self._catelog:
                if app and not data[cat]:
                    continue
                print()
                print("\n".join([f"- {k}: {v}"
                                 for k, v in data[cat].items()]))
            print("-" * 80)
            time.sleep(0)  # for unittest

        async def _query_forever():
            while True:
                data = await self._query(self.channels, executor=executor)
                _print_result(None, data)
                time.sleep(1.0)

        async def _read_forever():
            async for pid, data in self.aread():
                _print_result(pid, data)

        try:
            if correlate:
                asyncio.run(_read_forever())
            else:
                asyncio.run(_query_forever())
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
