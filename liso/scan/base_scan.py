"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import abc
from collections import deque, OrderedDict

import numpy as np

from .scan_param import JitterParam, SampleParam, StepParam
from ..logging import logger


class _BaseScan(abc.ABC):
    def __init__(self):
        self._params = OrderedDict()
        self._param_dists = OrderedDict()

    def _check_param_name(self, name):
        return name

    def _add_scan_param(self, name, *, dist=-1., **kwargs):
        name = self._check_param_name(name)

        if name in self._params:
            raise ValueError(f"Parameter {name} already exists!")

        try:
            param = StepParam(name, **kwargs)
        except TypeError:
            try:
                param = SampleParam(name, **kwargs)
            except TypeError:
                param = JitterParam(name, **kwargs)

        self._params[name] = param
        self._param_dists[name] = dist

    def _sort_param_sequence(self, seq):
        tol = tuple(self._param_dists.values())

        def _check_distance(a, b):
            for ia, ib, it in zip(a, b, tol):
                if abs(ia - ib) < it:
                    return False
            return True

        cache = list()
        ret_queue = deque()
        for item in zip(*seq):
            if not ret_queue:
                ret_queue.append(item)
            else:
                if _check_distance(item, ret_queue[-1]):
                    ret_queue.append(item)
                elif _check_distance(item, ret_queue[0]):
                    ret_queue.appendleft(item)
                else:
                    cache.append(item)
        n_pulses = len(cache) + len(ret_queue)

        for item in cache:
            length = len(ret_queue)
            for i in range(2, length-2, 2):
                if _check_distance(item, ret_queue[i-1]) and \
                        _check_distance(item, ret_queue[i]):
                    ret_queue.insert(i, item)
                    break

        if len(ret_queue) != n_pulses:
            return

        return list(ret_queue)

    def _generate_param_sequence(self, cycles):
        if not self._params:
            return []

        repeats = np.prod([len(param) for param in self._params.values()])
        ret = []
        for param in self._params.values():
            repeats = int(repeats / len(param))
            ret.append(param.generate(repeats=repeats, cycles=cycles))
            cycles *= len(param)

        for i in range(5):
            logger.debug(f"Generating scan parameter sequence (attempt {i+1})")
            seq = self._sort_param_sequence(ret)
            if seq is not None:
                return seq

        raise RuntimeError(
            "Failed to a parameter sequence with enough distance!")

    def summarize(self):
        text = '\n' + '=' * 80 + '\n'
        text += 'Scanned parameters:\n'
        text += self._summarize_parameters()
        text += '=' * 80 + '\n'
        return text

    def _summarize_parameters(self):
        scan_params = []
        sample_params = []
        jitter_params = []
        for param in self._params.values():
            if isinstance(param, StepParam):
                scan_params.append(param)
            elif isinstance(param, SampleParam):
                sample_params.append(param)
            elif isinstance(param, JitterParam):
                jitter_params.append(param)

        text = ''
        for params in (scan_params, sample_params, jitter_params):
            if params:
                text += "\n"
            for i, ele in enumerate(params):
                if i == 0:
                    text += ele.__str__()
                else:
                    text += ele.list_item()
        return text

    @abc.abstractmethod
    def _create_output_dir(self, parent):
        pass
