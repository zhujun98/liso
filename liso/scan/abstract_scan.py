"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import abc
from collections import deque, OrderedDict
import functools
import sys
import traceback
from typing import Awaitable, Callable, List

import numpy as np

from .scan_param import JitterParam, SampleParam, StepParam
from ..exceptions import LisoRuntimeError
from ..logging import logger


class AbstractScan(abc.ABC):
    def __init__(self):
        self._params = OrderedDict()
        self._param_dists = OrderedDict()

    def _parse_param_name(self, name: str) -> str:  # pylint: disable=no-self-use
        return name

    def add_param(self, name: str, *, dist=-1., **kwargs):
        """Add a parameter for scan.

        The kwargs will be passed to the construct of a ScanParam subclass.

        :param name: Parameter name in the simulation input file.
        :param kwargs: Keyword arguments will be passed to the constructor
            of the appropriate :class:`liso.scan.scan_param.ScanParam`.
        """
        name = self._parse_param_name(name)

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

    @staticmethod
    def _check_distance(a, b, tol):
        for ia, ib, it in zip(a, b, tol):
            if abs(ia - ib) < it:
                return False
        return True

    def _sort_param_sequence(self, seq):  # pylint: disable=inconsistent-return-statements
        tol = tuple(self._param_dists.values())

        cache = []
        ret_queue = deque()
        for item in zip(*seq):
            if not ret_queue:
                ret_queue.append(item)
            else:
                if self._check_distance(item, ret_queue[-1], tol):
                    ret_queue.append(item)
                elif self._check_distance(item, ret_queue[0], tol):
                    ret_queue.appendleft(item)
                else:
                    cache.append(item)
        n_pulses = len(cache) + len(ret_queue)

        for item in cache:
            length = len(ret_queue)
            for i in range(2, length-2, 2):
                if self._check_distance(item, ret_queue[i-1], tol) and \
                        self._check_distance(item, ret_queue[i], tol):
                    ret_queue.insert(i, item)
                    break

        if len(ret_queue) != n_pulses:
            return

        return list(ret_queue)

    def _generate_param_sequence(self, cycles: int) -> List[tuple]:
        """Generate a sequence of parameter combinations for scan.

        :raises ValueError: If generation of parameter sequence fails.
        """
        if not self._params:
            return []

        repeats = np.prod([len(param) for param in self._params.values()])
        ret = []
        for param in self._params.values():
            repeats = int(repeats / len(param))
            ret.append(param.generate(repeats=repeats, cycles=cycles))
            cycles *= len(param)

        for i in range(5):
            logger.debug("Generating scan parameter sequence (attempt %s)", i+1)
            seq = self._sort_param_sequence(ret)
            if seq is not None:
                return seq

        raise ValueError(
            "Failed to a parameter sequence with enough distance!")

    def summarize(self):
        text = '\n' + '=' * 120 + '\n'
        text += 'Scanned parameters:\n'
        text += self._summarize_parameters()
        text += '=' * 120 + '\n'
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
    def scan(self, *args, **kwargs) -> None:
        """Run the scan."""


def _capture_scan_error(f: Callable[..., Awaitable[None]]):
    @functools.wraps(f)
    async def decorated_f(*args, **kwargs):
        try:
            await f(*args, **kwargs)
        except LisoRuntimeError as e:
            _, _, exc_traceback = sys.exc_info()
            logger.debug(
                "%s, %s",
                repr(traceback.format_tb(exc_traceback)),
                str(e))
            logger.warning(str(e))
        except Exception as e:
            _, _, exc_traceback = sys.exc_info()
            logger.error(
                "%s, %s",
                repr(traceback.format_tb(exc_traceback)),
                str(e))
            raise
    return decorated_f
