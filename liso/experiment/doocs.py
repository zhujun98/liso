"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import abc
import asyncio
from collections import namedtuple, OrderedDict
from typing import Optional, Tuple

import numpy as np

from pydantic import (
    BaseModel, StrictBool, StrictFloat, StrictInt, conint, confloat, validator
)

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

_machine_event_loop = asyncio.get_event_loop()


class DoocsWriter:

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


class DoocsReader:

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


class NDArrayMeta(type):
    def __getitem__(self):
        return type('NDArray', (NDArray,))


class NDArray(np.ndarray, metaclass=NDArrayMeta):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def __modify_schema__(cls, field_schema):
        # __modify_schema__ should mutate the dict it receives in place,
        # the returned value will be ignored
        field_schema.update(
            type='NDArray',
        )

    @classmethod
    def validate_type(cls, v):
        if not isinstance(v, np.ndarray):
            raise TypeError('Input must be a numpy.ndarray')
        return v


class DoocsChannel(BaseModel, metaclass=abc.ABCMeta):
    address: str

    @validator('address')
    def doocs_address(cls, v):
        # An address must contain four fields:
        #
        fields = [field.strip() for field in v.split('/')]
        if len(fields) != 4 or not all(fields):
            raise ValueError("An address must have the form: "
                             "facility/device/location/property")
        return v

    def value_schema(self):
        """Return the value schema of the instance."""
        schema = self.__class__.schema()['properties']['value'].copy()
        schema.pop('title')
        return schema

    class Config:
        validate_assignment = True


class BoolDoocsChannel(DoocsChannel):
    value: StrictBool = False

    class Config:
        @staticmethod
        def schema_extra(schema, model):
            schema['properties']['value']['type'] = '|b1'


class Int64DoocsChannel(DoocsChannel):
    value: StrictInt = 0

    class Config:
        @staticmethod
        def schema_extra(schema, model):
            schema['properties']['value']['type'] = '<i8'


class UInt64DoocsChannel(DoocsChannel):
    value: conint(strict=True, ge=0) = 0

    class Config:
        @staticmethod
        def schema_extra(schema, model):
            schema['properties']['value']['type'] = '<u8'


class Int32DoocsChannel(DoocsChannel):
    value: conint(strict=True,
                  ge=np.iinfo(np.int32).min,
                  le=np.iinfo(np.int32).max) = 0

    class Config:
        @staticmethod
        def schema_extra(schema, model):
            schema['properties']['value']['type'] = '<i4'


class UInt32DoocsChannel(DoocsChannel):
    value: conint(strict=True,
                  ge=0,
                  le=np.iinfo(np.uint32).max) = 0
    class Config:
        @staticmethod
        def schema_extra(schema, model):
            schema['properties']['value']['type'] = '<u4'


class Int16DoocsChannel(DoocsChannel):
    value: conint(strict=True,
                  ge=np.iinfo(np.int16).min,
                  le=np.iinfo(np.int16).max) = 0

    class Config:
        @staticmethod
        def schema_extra(schema, model):
            schema['properties']['value']['type'] = '<i2'


class UInt16DoocsChannel(DoocsChannel):
    value: conint(strict=True,
                  ge=0,
                  le=np.iinfo(np.uint16).max) = 0

    class Config:
        @staticmethod
        def schema_extra(schema, model):
            schema['properties']['value']['type'] = '<u2'


class Float64DoocsChannel(DoocsChannel):
    value: StrictFloat = 0.0

    class Config:
        @staticmethod
        def schema_extra(schema, model):
            schema['properties']['value']['type'] = '<f8'


class Float32DoocsChannel(DoocsChannel):
    value: confloat(strict=True,
                    ge=np.finfo(np.float32).min,
                    le=np.finfo(np.float32).max) = 0

    class Config:
        @staticmethod
        def schema_extra(schema, model):
            schema['properties']['value']['type'] = '<f4'


class ImageDoocsChannel(DoocsChannel):
    shape: Tuple[int, int]
    # The array-protocol typestring, e.g. <i8, <f8, etc.
    dtype: str
    value: Optional[NDArray] = None

    @validator("dtype")
    def check_dtype(cls, v):
        """Check whether the input can be converted to a valid numpy.dtype.

        :param str v: dtype string. Must be valid to construct a data type
            object. For more details, check
            https://numpy.org/doc/stable/reference/arrays.dtypes.html.
        """
        # raise: TypeError
        dtype = np.dtype(v)
        return dtype.str

    @validator("value", always=True)
    def check_value(cls, v, values):
        if 'shape' not in values or 'dtype' not in values:
            # ValidationError will be raised later
            return v

        shape, dtype = values['shape'], values['dtype']

        if v is None:
            return np.zeros(shape=shape, dtype=dtype)

        if v.dtype.str != dtype:
            raise TypeError(
                f'Array dtypes do not match: {dtype} and {v.dtype.name}')

        if v.shape != shape:
            raise ValueError(
                f"Data shapes do not match: {shape} and {v.shape}")

        return v

    def value_schema(self):
        """Override."""
        schema = self.__class__.schema()['properties']['value'].copy()
        schema.pop('title')
        data = self.dict()
        schema['shape'] = data['shape']
        schema['dtype'] = data['dtype']
        return schema


_DoocsChannelFactory = namedtuple(
    "DoocsChannelFactory",
    ["BOOL",
     "INT64", "LONG", "UINT64", "ULONG",
     "INT32", "INT", "UINT32", "UINT",
     "INT16", "UINT16",
     "FLOAT64", "DOUBLE", "FLOAT32", "FLOAT",
     "IMAGE"]
)

doocs_channels = _DoocsChannelFactory(
    BOOL=BoolDoocsChannel,
    INT64=Int64DoocsChannel,
    LONG=Int64DoocsChannel,
    UINT64=UInt64DoocsChannel,
    ULONG=UInt64DoocsChannel,
    INT32=Int32DoocsChannel,
    INT=Int32DoocsChannel,
    UINT32=UInt32DoocsChannel,
    UINT=UInt32DoocsChannel,
    INT16=Int16DoocsChannel,
    UINT16=UInt16DoocsChannel,
    FLOAT64=Float64DoocsChannel,
    DOUBLE=Float64DoocsChannel,
    FLOAT32=Float32DoocsChannel,
    FLOAT=Float32DoocsChannel,
    IMAGE=ImageDoocsChannel,
)
