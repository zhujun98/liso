from datetime import datetime

import numpy as np


class DoocsDataGenerator:
    @staticmethod
    def data_type(dtype):
        if dtype == "any":
            return 'ANY'
        if np.dtype(dtype) == np.float32:
            return 'FLOAT'
        if np.dtype(dtype) == np.float64:
            return 'DOUBLE'
        if np.dtype(dtype) == np.int32:
            return 'INT'
        if np.dtype(dtype) == np.int64:
            return 'LONG'
        if np.dtype(dtype) == np.bool:
            return 'BOOL'
        raise ValueError

    @classmethod
    def scalar(cls, v, schema, *, pid):
        return {
            'data': v,
            'macropulse': pid,
            'miscellaneous': {},
            'timestamp': datetime.timestamp(datetime.now()),
            'type': cls.data_type(schema['type'])
        }

    @classmethod
    def array(cls, schema, *, pid):
        shape = schema['shape']
        if len(shape) == 1:
            a_type = 'ARRAY'  # FIXME
        elif len(shape) == 2:
            if shape[1] == 2:
                a_type = "SPECTRUM"
            else:
                a_type = "IMAGE"
        else:
            raise ValueError

        return {
            'data': np.ones(schema['shape'], dtype=schema['dtype']),
            'macropulse': pid,
            'miscellaneous': {},
            'timestamp': datetime.timestamp(datetime.now()),
            'type': a_type,
        }
