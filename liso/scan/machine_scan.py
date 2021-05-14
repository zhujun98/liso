"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import abc
import asyncio
from collections import deque, OrderedDict
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import pathlib
import re
import sys
import traceback

import numpy as np

from .base_scan import _BaseScan
from ..exceptions import LisoRuntimeError
from ..io import ExpWriter
from ..logging import logger


class MachineScan(_BaseScan):
    """Class for performing scans with a real machine."""

    def __init__(self, machine):
        """Initialization.

        :param _BaseMachine machine: Machine instance.
        """
        super().__init__()

        self._machine = machine

        self._param_readouts = dict()

    def _create_output_dir(self, parent):
        parent_path = pathlib.Path(parent)
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

    def scan(self, cycles=1, output_dir="./", *,
             tasks=None,
             chmod=True,
             timeout=None,
             group=1,
             seed=None):
        """Start a parameter scan.

        :param int cycles: number of cycles of the parameter space. For
            pure jitter study, it is the number of runs since the size
            of variable space is 1.
        :param str output_dir: Directory in which data for each run is
            stored in in its own sub-directory.
        :param int/None tasks: maximum number of concurrent tasks for
            read and write.
        :param bool chmod: True for changing the permission to 400 after
            finishing writing.
        :param float/None timeout: timeout when correlating data by macropulse
            ID, in seconds.
        :param int group: writer group.
        :param int/None seed: seed for the legacy MT19937 BitGenerator
            in numpy.
        """
        if tasks is None:
            tasks = multiprocessing.cpu_count()
        executor = ThreadPoolExecutor(max_workers=tasks)

        try:
            ret = self._machine.take_snapshot(self._params)
            logger.info(f"Current values of the scanned parameters: "
                        f"{str(ret)[1:-1].replace(': ', ' = ')}")
        except LisoRuntimeError:
            raise RuntimeError("Failed to read all the initial values of "
                               "the scanned parameters!")

        logger.info(f"Starting parameter scan with {tasks} CPUs.")
        logger.info(self.summarize())

        np.random.seed(seed)

        output_dir = self._create_output_dir(output_dir)

        sequence = self._generate_param_sequence(cycles)
        n_pulses = len(sequence) if sequence else cycles
        with ExpWriter(output_dir,
                       schema=self._machine.schema,
                       chmod=chmod,
                       group=group) as writer:
            count = 0
            while count < n_pulses:
                mapping = dict()
                if sequence:
                    for i, k in enumerate(self._params):
                        mapping[k] = {'value': sequence[count][i]}
                        mapping[k].update(self._param_readouts[k])
                count += 1
                logger.info(f"Scan {count:06d}: "
                            + str({address: item['value']
                                   for address, item in mapping.items()})
                            [1:-1].replace(': ', ' = '))

                try:
                    idx, controls, diagnostics = self._machine.write_and_read(
                        executor=executor,
                        mapping=mapping,
                        timeout=timeout,
                    )
                    writer.write(idx, controls, diagnostics)
                except LisoRuntimeError as e:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    logger.debug(repr(traceback.format_tb(exc_traceback))
                                 + str(e))
                    logger.warning(str(e))
                except Exception as e:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    logger.error(
                        f"(Unexpected exceptions): "
                        + repr(traceback.format_tb(exc_traceback))
                        + str(e))
                    raise

        logger.info(f"Scan finished!")

    def add_param(self, name, readout=None, tol=1e-6, **kwargs):
        """Add a parameter for scan.

        The kwargs will be passed to the construct of a ScanParam subclass.

        :param str name: the DOOCS address.
        :param str/None readout: the DOOCS address for validating the value
            being written. If None, the written value will not be validated.
        :param float tol: tolerance for the validation. Positive value for
            absolute error and negative value for relative error.
        """
        self._add_scan_param(name, **kwargs)

        self._param_readouts[name] = {'readout': readout}

        self._param_readouts[name]['tol'] = tol
