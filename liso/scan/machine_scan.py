"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import enum
import multiprocessing
from pathlib import Path
import time
from typing import Optional, Tuple, Union

import numpy as np

from .abstract_scan import AbstractScan
from ..exceptions import LisoRuntimeError
from ..io import ExpWriter
from ..logging import logger
from ..experiment.machine_interface import MachineInterface


class ScanPolicy(enum.Enum):
    READ_AFTER_DELAY = 1


class MachineScan(AbstractScan):
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

    def _touch(self,
               mapping: dict,
               loop: asyncio.AbstractEventLoop,
               executor: ThreadPoolExecutor) -> Tuple[int, dict]:
        """Touch the machine with reading after writing."""
        self._interface.write(mapping, loop=loop, executor=executor)
        if self._policy == ScanPolicy.READ_AFTER_DELAY:
            time.sleep(self._read_delay)
        idx, data = self._interface.read(
            loop=loop, executor=executor)

        ret = dict()
        for key, item in data.items():
            ret[key] = {k: v['data'] for k, v in item.items()}
        return idx, ret

    def _scan_imp(self, sequence: list,
                  writer: ExpWriter,
                  loop: asyncio.AbstractEventLoop,
                  executor: ThreadPoolExecutor) -> None:
        n_pulses = len(sequence)

        count = 0
        while count < n_pulses:
            mapping = dict()
            for i, k in enumerate(self._params):
                mapping[k] = sequence[count][i]
            count += 1
            logger.info(
                "Scan %06d: %s",
                count,
                str({address: value for address, value
                     in mapping.items()})[1:-1].replace(': ', ' = '))

            self._collect_result(
                writer, self._touch, mapping, loop, executor)

    def scan(self, cycles: int = 1, *,  # pylint: disable=arguments-differ
             n_tasks: Optional[int] = None,
             output_dir: Union[str, Path] = "./",
             chmod: bool = True,
             group: int = 1,
             seed: Optional[int] = None):
        """Run the scan.

        :param cycles: Number of cycles of the parameter space. For
            pure jitter study, it is the number of runs since the size
            of variable space is 1.
        :param n_tasks: Maximum number of concurrent tasks for
            read and write.
        :param output_dir: Directory in which data for each run is
            stored in in its own sub-directory.
        :param chmod: True for changing the permission to 400 after
            finishing writing.
        :param group: Writer group.
        :param seed: Seed for the legacy MT19937 BitGenerator in numpy.

        :raises ValueError: If generation of parameter sequence fails.
        """
        if not self._params:
            raise ValueError("No scan parameters specified!")

        if n_tasks is None:
            n_tasks = multiprocessing.cpu_count()

        try:
            _, ret = self._interface.read()
            # TODO: improve
            logger.info("Current values of the scanned parameters:\n %s", ret)
        except LisoRuntimeError as e:
            raise RuntimeError("Failed to read all the initial values of "
                               "the scanned parameters!") from e

        output_dir = self._create_output_dir(output_dir)

        sequence = self._generate_param_sequence(cycles)

        logger.info("Starting parameter scan with %s CPUs. "
                    "Scan result will be save at %s",
                    n_tasks, output_dir.resolve())
        logger.info(self.summarize())

        with ExpWriter(output_dir,
                       schema=self._interface.schema,
                       chmod=chmod,
                       group=group) as writer:
            np.random.seed(seed)
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=n_tasks)
            self._scan_imp(sequence, writer, loop, executor)

        logger.info("Scan finished!")
