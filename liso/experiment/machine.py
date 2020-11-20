"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from pydantic import ValidationError

try:
    from pydoocs import read as pydoocs_read
    from pydoocs import write as pydoocs_write
    from pydoocs import DoocsException, PyDoocsException
except ModuleNotFoundError:
    __pydoocs_error_msg = "pydoocs is required to communicate with a real " \
                          "machine using DOOCS control system!"
    def pydoocs_read(*args):
        raise ModuleNotFoundError(__pydoocs_error_msg)
    def pydoocs_write(*args):
        raise ModuleNotFoundError(__pydoocs_error_msg)
    class DoocsException(Exception):
        pass
    class PyDoocsException(Exception):
        pass

from ..exceptions import LisoRuntimeError
from ..logging import logger
from ..utils import profiler

_machine_event_loop = asyncio.get_event_loop()


class _DoocsWriter:

    _DELAY_EXCEPTION = 0.1

    def __init__(self):
        super().__init__()

    async def _write_channel(self, address, value, *, delay=0., executor=None):
        if delay > 0.:
            await asyncio.sleep(delay)
        return await _machine_event_loop.run_in_executor(
            executor, pydoocs_write, address, value)

    def _get_result(self, address, task):
        try:
            task.result()
            return True
        except ModuleNotFoundError as e:
            logger.error(repr(e))
            raise
        except (DoocsException, PyDoocsException) as e:
            logger.warning(f"Failed to write to {address}: {repr(e)}")
        except Exception as e:
            logger.error(f"Unexpected exception when writing to"
                         f" {address}: {repr(e)}")
            # FIXME: here should raise

        return False

    async def write_channels(self, executor, writein, *, attempts=5):
        if not writein:
            return True

        _DELAY_EXCEPTION = self._DELAY_EXCEPTION

        future_ret = {asyncio.create_task(
            self._write_channel(addr, v, executor=executor)): (addr, v)
                      for addr, v in writein.items()}

        for i in range(attempts):
            done, _ = await asyncio.wait(
                future_ret, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                address, value = future_ret[task]

                if not self._get_result(address, task):
                    future_ret[asyncio.create_task(self._write_channel(
                        address, value,
                        executor=executor,
                        delay=_DELAY_EXCEPTION))] = (address, value)

                del future_ret[task]
                if not future_ret:
                    return

        raise LisoRuntimeError(
            "Failed to write new values to all channels!")


class _DoocsReader:

    _NO_EVENT = 0
    _DELAY_NO_EVENT = 1.
    _DELAY_STALE = 0.2
    _DELAY_EXCEPTION = 0.1

    def __init__(self):
        super().__init__()

        self._channels = set()
        self._no_event = set()

        self._last_correlated = 0

    def _compare_readout(self, data, expected):
        for address, (v, tol) in expected.items():
            if abs(data[address] - v) > tol:
                return False, \
                       f"{address} - expected: {v}, actual: {data[address]}"
        return True, ""

    async def _read_channel(self, address, *, delay=0., executor=None):
        if delay > 0.:
            await asyncio.sleep(delay)
        return await _machine_event_loop.run_in_executor(
            executor, pydoocs_read, address)

    async def read_channels(self, addresses, *, executor=None, attempts=3):
        future_ret = {asyncio.create_task(
            self._read_channel(address, executor=executor)): address
            for address in addresses}

        ret = dict()
        for i in range(attempts):
            done, _ = await asyncio.wait(future_ret)

            for task in done:
                address = future_ret[task]
                data = self._get_result(address, task)
                if data is not None:
                    ret[address] = data
                else:
                    future_ret[asyncio.create_task(self._read_channel(
                        address, executor=executor))] = address
                del future_ret[task]

            if not future_ret:
                return ret

        raise LisoRuntimeError(f"Failed to read data from "
                               f"{list(future_ret.values())}")

    def _get_result(self, address, task):
        try:
            return task.result()
        except ModuleNotFoundError as e:
            logger.error(repr(e))
            raise
        except (DoocsException, PyDoocsException) as e:
            logger.warning(f"Failed to read data from {address}: {repr(e)}")
        except Exception as e:
            logger.error(f"Unexpected exception when writing to "
                         f"{address}: {repr(e)}")

    async def correlate(self, executor, readout, *, timeout):
        n_events = len(self._channels) - len(self._no_event)
        cached = OrderedDict()

        _NO_EVENT = self._NO_EVENT
        _DELAY_NO_EVENT = self._DELAY_NO_EVENT
        _DELAY_STALE = self._DELAY_STALE
        _DELAY_EXCEPTION = self._DELAY_EXCEPTION
        _SENTINEL = object()
        correlated = dict()

        future_ret = {asyncio.create_task(
            self._read_channel(address, executor=executor)): address
                 for address in self._channels if address not in self._no_event}
        future_ret[asyncio.create_task(asyncio.sleep(timeout))] = _SENTINEL

        running = True
        while running:
            done, _ = await asyncio.wait(
                future_ret, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                address = future_ret[task]
                ch_data = self._get_result(address, task)

                if address is _SENTINEL:
                    running = False
                    continue

                delay = 0.
                # exception not raised in _get_result
                if ch_data is not None:
                    pid = ch_data['macropulse']
                    if pid > self._last_correlated:
                        if pid not in cached:
                            cached[pid] = dict()
                        cached[pid][address] = ch_data['data']

                        if len(cached[pid]) == n_events:
                            compare_ret, msg = self._compare_readout(
                                cached[pid], readout)
                            if not compare_ret:
                                logger.debug(
                                    f"The newly written channels have not "
                                    f"all taken effect: {msg}")
                                # remove old data
                                for key in list(cached.keys()):
                                    if key > pid:
                                        break
                                    del cached[key]
                                continue

                            no_event_data = await self.read_channels(
                                self._no_event)
                            for ne_addr, ne_item in no_event_data.items():
                                correlated[ne_addr] = ne_item['data']

                            logger.info(
                                f"Correlated {len(self._channels)}"
                                f"({n_events}) channels with "
                                f"macropulse ID: {pid}")

                            self._last_correlated = pid
                            correlated.update(cached[pid])

                            return pid, correlated
                    elif pid == _NO_EVENT:
                        if address not in correlated:
                            n_events -= 1
                        correlated[address] = ch_data['data']

                        delay = _DELAY_NO_EVENT
                    else:
                        if pid < 0:
                            # TODO: document when a macropulse ID is -1
                            logger.warning(
                                f"Received data from channel {address} "
                                f"with invalid macropulse ID: {pid}")

                        delay = _DELAY_STALE
                else:
                    delay = _DELAY_EXCEPTION

                del future_ret[task]
                future_ret[asyncio.create_task(self._read_channel(
                    address, executor=executor, delay=delay))] = address

        raise LisoRuntimeError("Unable to match all data!")

    def add_channel(self, address, no_event=False):
        self._channels.add(address)
        if no_event:
            self._no_event.add(address)


class _DoocsMachine:
    """Base class for machine interface using DOOCS control system."""

    _facility_name = None

    def __init__(self):
        self._controls = OrderedDict()
        self._diagnostics = OrderedDict()

        self._reader = _DoocsReader()
        self._writer = _DoocsWriter()

    @property
    def channels(self):
        """Return a list of all DOOCS channels."""
        return list(self._controls) + list(self._diagnostics)

    @property
    def controls(self):
        """Return a list of DOOCS channels for control data."""
        return list(self._controls)

    @property
    def diagnostics(self):
        """Return a list of DOOCS channels for diagnostic data."""
        return list(self._diagnostics)

    @property
    def schema(self):
        """Return the schema of all DOOCS channels."""
        return ({k: v.value_schema() for k, v in self._controls.items()},
                {k: v.value_schema() for k, v in self._diagnostics.items()})

    def _check_address(self, address):
        if address in self._controls or address in self._diagnostics:
            raise ValueError(f"{address} already exists!")

        if not address.startswith(self._facility_name):
            raise ValueError(f"{address} must start with {self._facility_name}")

    def add_control_channel(self, kls, address, *, no_event=False, **kwargs):
        """Add a DOOCS channel for control data.

        :param DoocsChannel kls: a concrete DoocsChannel class.
        :param str address: DOOCS address.
        :param bool no_event: True for a non-event-based channel.
        **kwargs: keyword arguments which will be passed to the constructor
            of kls after address.

        Examples:
            from liso import doocs_channels as dc
            from liso import EuXFELInterface

            m = EuXFELInterface()
            m.add_control_channel(
                dc.FLOAT32, 'XFEL.RF/LLRF.CONTROLLER/CTRL.AH1.I1/SP.PHASE')
        """
        self._check_address(address)
        self._controls[address] = kls(address=address, **kwargs)
        self._reader.add_channel(address, no_event)

    def add_diagnostic_channel(self, kls, address, *, no_event=False, **kwargs):
        """Add a DOOCS channel to diagnostic data.

        :param DoocsChannel kls: a concrete DoocsChannel class.
        :param str address: DOOCS address.
        :param bool no_event: True for a non-event-based channel.
        **kwargs: keyword arguments which will be passed to the constructor
            of kls after address.

        Examples:
            from liso import doocs_channels as dc
            from liso import EuXFELInterface

            m = EuXFELInterface()
            m.add_diagnostic_channel(
                dc.IMAGE, 'XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ',
                shape=(1750, 2330), dtype='uint16')
        """
        self._check_address(address)
        self._diagnostics[address] = kls(address=address, **kwargs)
        self._reader.add_channel(address, no_event)

    def _compile(self, mapping):
        writein = dict()
        readout = dict()
        if mapping is not None:
            for address, item in mapping.items():
                writein[address] = item['value']
                if item.get('readout', None) is not None:
                    readout_address = item['readout']
                    if readout_address not in self._controls \
                            and readout_address not in self._diagnostics:
                        raise ValueError(f"Channel {readout_address} has not "
                                         f"been registered!")

                    readout[readout_address] = (item['value'], item['tol'])

        return writein, readout

    def _update_channels(self, correlated):
        control_data = dict()
        for address, ch in self._controls.items():
            ch.value = correlated[address]  # validate
            control_data[address] = ch.value

        diagnostic_data = dict()
        for address, ch in self._diagnostics.items():
            ch.value = correlated[address]  # validate
            diagnostic_data[address] = ch.value

        return control_data, diagnostic_data

    async def _write_read_once(self, mapping, *, executor, timeout):
        writein, readout = self._compile(mapping)
        await self._writer.write_channels(executor, writein)
        return await self._reader.correlate(executor, readout, timeout=timeout)

    @profiler("machine write and read")
    def write_and_read(self, *, mapping=None, executor=None, timeout=None):
        """Write and read the machine once.

        :param dict mapping: a dictionary with keys being the DOOCS channel
            addresses and values being the numbers to be written into the
            corresponding address.
        :param ThreadPoolExecutor executor: a ThreadPoolExecutor object.
        :param float/None timeout: timeout when correlating data by macropulse
            ID, in seconds. If None, it is set to the default value 2.0.

        :raises:
            LisoRuntimeError: if validation fails or it is unable to
                correlate data.
        """
        if executor is None:
            executor = ThreadPoolExecutor()
        if timeout is None:
            timeout = 2.0

        pid, correlated = _machine_event_loop.run_until_complete(
            self._write_read_once(mapping, executor=executor, timeout=timeout))

        try:
            control_data, diagnostic_data = self._update_channels(correlated)
        except ValidationError as e:
            raise LisoRuntimeError(repr(e))

        return pid, control_data, diagnostic_data

    def take_snapshot(self, channels):
        if not channels:
            return

        return {address: data['data']
                for address, data in _machine_event_loop.run_until_complete(
                self._reader.read_channels(channels)).items()}


class EuXFELInterface(_DoocsMachine):
    _facility_name = 'XFEL'


class FLASHInterface(_DoocsMachine):
    _facility_name = 'FLASH'
