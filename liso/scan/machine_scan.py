"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import enum
import multiprocessing
import pathlib
import re
import sys
import time
import traceback
from typing import Optional

import numpy as np

from .base_scan import BaseScan
from ..exceptions import LisoRuntimeError
from ..io import ExpWriter
from ..logging import logger
from ..experiment.machine_interface import MachineInterface


class ScanPolicy(enum.Enum):
    READ_AFTER_DELAY = 1


class MachineScan(BaseScan):
    """Class for performing scans with a real machine."""

    def __init__(self, interface: MachineInterface,
                 policy: ScanPolicy = ScanPolicy.READ_AFTER_DELAY,
                 read_delay: float = 1.0) -> None:
        """Initialization.

        :param interface: MachineInterface instance.
        :param policy: Policy about how the scan is performed.
        :param read_delay: Delay for reading channel data in seconds after
            writing channels. Use only when
            policy = ScanPolicy.READ_AFTER_DELAY.
        """
        super().__init__()

        self._interface = interface

        if not isinstance(policy, ScanPolicy):
            raise ValueError(f"{policy} is not a valid scan policy! "
                             f"Valid values are {[str(p) for p in ScanPolicy]}")
        self._policy = policy
        self._read_delay = read_delay

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

    def scan(self, cycles: int = 1, output_dir: str = "./", *,
             tasks: Optional[int] = None,
             chmod: bool = True,
             timeout: float = None,
             group: int = 1,
             seed: Optional[int] = None):
        """Start a parameter scan.

        :param cycles: Number of cycles of the parameter space. For
            pure jitter study, it is the number of runs since the size
            of variable space is 1.
        :param output_dir: Directory in which data for each run is
            stored in in its own sub-directory.
        :param tasks: Maximum number of concurrent tasks for
            read and write.
        :param chmod: True for changing the permission to 400 after
            finishing writing.
        :param timeout: Timeout when correlating data by macropulse
            ID, in seconds.
        :param group: Writer group.
        :param seed: Seed for the legacy MT19937 BitGenerator in numpy.
        """
        if not self._params:
            raise ValueError("No scan parameters specified!")

        if tasks is None:
            tasks = multiprocessing.cpu_count()
        executor = ThreadPoolExecutor(max_workers=tasks)
        loop = asyncio.get_event_loop()

        try:
            ret = self._interface.read()
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
                       schema=self._interface.schema,
                       chmod=chmod,
                       group=group) as writer:
            count = 0
            while count < n_pulses:
                mapping = dict()
                if sequence:
                    for i, k in enumerate(self._params):
                        mapping[k] = {'value': sequence[count][i]}
                count += 1
                logger.info(f"Scan {count:06d}: "
                            + str({address: item['value']
                                   for address, item in mapping.items()})
                            [1:-1].replace(': ', ' = '))

                try:
                    self._interface.write(mapping, loop=loop, executor=executor)
                    if self._policy == ScanPolicy.READ_AFTER_DELAY:
                        time.sleep(self._read_delay)
                    idx, controls, diagnostics = self._interface.read(
                        loop=loop, executor=executor)

                    c_data = {k: v['data'] for k, v in controls.items()}
                    d_data = {k: v['data'] for k, v in diagnostics.items()}

                    writer.write(idx, c_data, d_data)
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
