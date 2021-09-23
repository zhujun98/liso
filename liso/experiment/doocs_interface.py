"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import asyncio
from collections import deque, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
import random
import time
from typing import (
    Any, Callable, Dict, List, Optional, FrozenSet, Tuple, Type, Union
)

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
from .doocs_channels import AnyDoocsChannel, DoocsChannel


class Correlator:
    def __init__(self, *,
                 timeout: float,
                 retry_after: float):
        self._timeout = timeout
        self._retry_after = retry_after

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

    async def _collect_event(self,
                             channels: set,
                             buffer: OrderedDict,
                             ready: deque, *,
                             query) -> None:
        tasks = {
            asyncio.create_task(query(address)): address
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

                        if len(buffer[pid]) == len(channels):
                            ready.append(pid)
                            self._last_correlated = pid

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
                    tasks[asyncio.create_task(query(
                        address, delay=self._retry_after))] = address

        await self._cancel_all(tasks)

    async def _collect_non_event(self,
                                 channels: set,
                                 buffer: dict,
                                 ready: deque,
                                 *, query) -> None:
        if not channels:
            ready.append(object())
            return

        tasks = dict()

        tasks.update({
            asyncio.create_task(query(address)): address
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
                tasks[asyncio.create_task(query(
                    address, delay=self._retry_after))] = address

        await self._cancel_all(tasks)

    async def _timer(self, n: Optional[int]):
        if self._timeout is None or n is None:
            return

        await asyncio.sleep(self._timeout * n)
        self._running = False

    async def _aggregate(self, n: Optional[int], event, event_ready,
                         non_event, non_event_ready, *,
                         callback: Optional[Callable] = None):
        ret = []
        while self._running:  # pylint: disable=too-many-nested-blocks
            if non_event_ready:
                while True:
                    try:
                        pid = event_ready.popleft()
                        data = event.pop(pid)
                        data.update(non_event)

                        if callback is not None:
                            callback(pid, data)

                        if n is not None:
                            ret.append((pid, data))
                            if len(ret) == n:
                                self._running = False

                    except IndexError:
                        break

            await asyncio.sleep(self._retry_after)

        return ret

    async def correlate(self, n: Optional[int], event: set, non_event: set, *,
                        query: Callable, callback: Callable):
        """Correlate all the given channels.

        :param n:
        :param event:
        :param non_event:
        :param query:
        :param callback:
        """
        event_buffer = OrderedDict()
        event_ready = deque()
        non_event_buffer = dict()
        non_event_ready = deque()

        self._running = True
        ret = await asyncio.gather(
            asyncio.create_task(self._timer(n)),
            asyncio.create_task(self._collect_event(
                event, event_buffer, event_ready, query=query)),
            asyncio.create_task(self._collect_non_event(
                non_event, non_event_buffer, non_event_ready, query=query)),
            asyncio.create_task(self._aggregate(
                n, event_buffer, event_ready, non_event_buffer, non_event_ready,
                callback=callback))
        )

        return ret[-1]


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

        tc = config.get("timeout.correlation")
        irr = config.get("interval.read.retry")

        self._corr = Correlator(
            timeout=2.0 if tc is None else tc,
            retry_after=0.01 if irr is None else irr
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
            >>>     'XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ', dc.IMAGE,
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
            mapping_write = {self._control_write[k]: v
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

    def parse_readout(self, readout: dict, *,  # pylint: disable=arguments-differ
                      verbose: bool = True,
                      validate: bool = False) -> Dict[str, Any]:
        """Parse readout.

        :param readout: Readout data.
        :param verbose: True for keeping the whole DOOCS message. Otherwise,
            only the 'data' of the message will be kept.
        :param validate: True for validate the readout 'data'.

        :raises LisoRuntimeError: If validation fails.
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
                    except ValidationError as e:
                        raise LisoRuntimeError("Validation error") from e
            ret[cat] = sub_ret
        return ret

    @staticmethod
    async def _read_channel(address: str, *,
                            delay: float = 0,
                            loop: asyncio.AbstractEventLoop,
                            executor: ThreadPoolExecutor) -> Tuple[str, Optional[dict]]:
        """Read the data from a single channel.

        The returned data from each channel contains the following keys:
            data, macropulse, timestamp, type, miscellaneous
        """
        if delay > 0:
            await asyncio.sleep(delay)

        try:
            data = await loop.run_in_executor(executor, pydoocs_read, address)
            return address, data

        except ModuleNotFoundError as e:
            logger.error(repr(e))
            raise
        except (DoocsException, PyDoocsException) as e:
            logger.error("Failed to read data from %s: {%s}", address, repr(e))
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Unexpected exception when reading from "
                         "{%s}: {%s}", address, repr(e))
        return address, None

    async def _query(self,
                     loop: asyncio.AbstractEventLoop,
                     executor: ThreadPoolExecutor) -> Dict[str, dict]:
        tasks = [
            asyncio.create_task(self._read_channel(
                address, loop=loop, executor=executor))
            for address in self.channels
        ]

        ret = dict()
        for fut in asyncio.as_completed(tasks, timeout=self._timeout_read):
            address, data = await fut
            ret[address] = data

        return ret

    @profiler("DOOCS interface query")
    def query(self, *,
              loop: Optional[asyncio.AbstractEventLoop] = None,
              executor: Optional[ThreadPoolExecutor] = None) -> Dict[str, dict]:
        """Read data from channels once without correlating them.

        :param loop: The event loop.
        :param executor: ThreadPoolExecutor instance.

        :raises ModuleNotFoundError: If PyDOOCS cannot be imported.
        :raises LisoRuntimeError: If validation fails.
        """
        if executor is None:
            executor = ThreadPoolExecutor()
        if loop is None:
            loop = asyncio.get_event_loop()

        return loop.run_until_complete(self._query(loop, executor))

    @profiler("DOOCS interface read")
    def read(self, n: Optional[int] = None, *,  # pylint: disable=arguments-differ
             loop: Optional[asyncio.AbstractEventLoop] = None,
             executor: Optional[ThreadPoolExecutor] = None,
             callback: Optional[Callable] = None) -> List[Tuple[int, Dict[str, dict]]]:
        """Read correlated data from all the channels.

        :param n: Number of correlated pulses to return.
        :param loop: The event loop.
        :param executor: ThreadPoolExecutor instance.
        :param callback: Callback function

        :raises ModuleNotFoundError: If PyDOOCS cannot be imported.
        :raises LisoRuntimeError: If there is no event channel.
        """
        if executor is None:
            executor = ThreadPoolExecutor()
        if loop is None:
            loop = asyncio.get_event_loop()

        if not self._event:
            raise LisoRuntimeError("At least one event channel is required to "
                                   "provide macropulse ID.")

        return loop.run_until_complete(self._corr.correlate(
            n, self._event, self._non_event,
            query=partial(self._read_channel, loop=loop, executor=executor),
            callback=callback
        ))

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
        def _write(pid, data):
            data = self.parse_readout(data, verbose=False, validate=True)
            writer.write(pid, data)
            time.sleep(0.001)

        output_dir = create_next_run_folder(output_dir)

        logger.info("Starting acquiring data and saving data to %s",
                    output_dir.resolve())

        loop = asyncio.get_event_loop()
        with ExpWriter(output_dir,
                       schema=self.schema,
                       chmod=chmod,
                       group=group) as writer:
            try:
                self.read(loop=loop, executor=executor, callback=_write)
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
        :param verbose:
        :param app: True for used in app with all channels belong to
            diagnostic.
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

        loop = asyncio.get_event_loop()
        try:
            if correlate:
                self.read(loop=loop, executor=executor, callback=_print_result)
            else:
                while True:
                    _print_result(
                        None, self.query(loop=loop, executor=executor))
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
