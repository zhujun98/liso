"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import abc
from collections import namedtuple
from typing import Optional, Tuple

import numpy as np

from pydantic import (
    BaseModel, StrictBool, StrictFloat, StrictInt, conint, confloat, validator
)


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
