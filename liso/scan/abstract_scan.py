"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import abc
from collections import deque, OrderedDict
from pathlib import Path
import re
import sys
from typing import Callable
import traceback

import numpy as np

from .scan_param import JitterParam, SampleParam, StepParam
from ..exceptions import LisoRuntimeError
from ..io.writer import WriterBase
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

    def _generate_param_sequence(self, cycles: int) -> list:
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

    @staticmethod
    def _create_output_dir(parent: str) -> Path:
        """Maybe create a directory to store the output data."""
        parent_path = Path(parent)
        # It is allowed to use an existing parent directory,
        # but not a run folder.
        parent_path.mkdir(exist_ok=True)

        next_run_index = 1  # starting from 1
        for d in parent_path.iterdir():
            # Here d could also be a file
            if re.search(r'r\d{4}', d.name):
                seq = int(d.name[1:])
                if seq >= next_run_index:
                    next_run_index = seq + 1

        next_output_dir = parent_path.joinpath(f'r{next_run_index:04d}')
        next_output_dir.mkdir(parents=True, exist_ok=False)
        return next_output_dir

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
    def scan(self, *args, **kwargs) -> None:
        """Run the scan."""

    @staticmethod
    def _collect_result(writer: WriterBase, collector: Callable, *args) -> None:
        """Collect result and write into file."""
        try:
            writer.write(*collector(*args))
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
                "(Unexpected exceptions): %s, %s",
                repr(traceback.format_tb(exc_traceback)),
                str(e))
            raise
