from datetime import datetime

import numpy as np


class DoocsDataGenerator:
    @staticmethod
    def data_type(dtype):
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
    def image(cls, schema, *, pid):
        return {
            'data': np.ones(schema['shape'], dtype=schema['dtype']),
            'macropulse': pid,
            'miscellaneous': {},
            'timestamp': datetime.timestamp(datetime.now()),
            'type': 'IMAGE',
        }
