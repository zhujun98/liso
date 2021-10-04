"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import enum
import multiprocessing
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .abstract_scan import AbstractScan, _capture_scan_error
from ..io import create_next_run_folder, ExpWriter
from ..logging import logger
from ..exceptions import LisoRuntimeError
from ..experiment.machine_interface import MachineInterface


class ScanPolicy(enum.Enum):
    READ_AFTER_DELAY = 1


class MachineScan(AbstractScan):
    """Class for performing scans with a real machine."""

    def __init__(self, interface: MachineInterface,
                 policy: ScanPolicy = ScanPolicy.READ_AFTER_DELAY,
                 read_delay: float = 1.0,
                 n_reads: int = 1) -> None:
        """Initialization.

        :param interface: MachineInterface instance.
        :param policy: Policy about how the scan is performed.
        :param read_delay: Delay for reading channel data in seconds after
            writing channels. Use only when
            policy = ScanPolicy.READ_AFTER_DELAY.
        :param n_reads: Number of reads after each write.
        """
        super().__init__()

        self._interface = interface

        if not isinstance(policy, ScanPolicy):
            raise ValueError(f"{policy} is not a valid scan policy! "
                             f"Valid values are {[str(p) for p in ScanPolicy]}")
        self._policy = policy
        self._read_delay = read_delay

        if not isinstance(n_reads, int) and n_reads <= 0:
            raise ValueError(f"n_reads must be a positive integer: {n_reads}")
        self._n_reads = n_reads

    @_capture_scan_error
    async def _collect(self,
                       writer: ExpWriter,
                       mapping: dict,
                       executor: ThreadPoolExecutor) -> None:
        """Touch the machine with reading after writing."""
        await self._interface.awrite(mapping, executor=executor)

        if self._policy == ScanPolicy.READ_AFTER_DELAY:
            await asyncio.sleep(self._read_delay)

        items = []
        async for pid, data in self._interface.aread(
                self._n_reads, executor=executor):
            data = self._interface.parse_readout(
                data, verbose=False, validate=True)
            items.append((pid, data))

        if len(items) < self._n_reads:
            raise LisoRuntimeError(
                f"Failed to readout {self._n_reads} data points")

        for item in items:
            writer.write(*item)

    async def _scan(self, sequence: List[tuple], *,
                    executor: ThreadPoolExecutor,
                    output_dir: Path,
                    chmod: bool,
                    group: int) -> None:
        n_pulses = len(sequence)

        async with self._interface.safe_awrite(
                self._params.keys(), executor=executor):

            with ExpWriter(output_dir,
                           schema=self._interface.schema,
                           chmod=chmod,
                           group=group) as writer:
                count = 0
                while count < n_pulses:
                    mapping = OrderedDict()
                    for i, k in enumerate(self._params):
                        mapping[k] = sequence[count][i]
                    count += 1
                    logger.info(
                        "Scan %06d: %s", count, [v for v in mapping.values()])

                    await self._collect(writer, mapping, executor)

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

        executor = ThreadPoolExecutor(max_workers=n_tasks)

        output_dir = create_next_run_folder(output_dir)

        np.random.seed(seed)
        sequence = self._generate_param_sequence(cycles)

        logger.info("Starting parameter scan with %s CPUs. "
                    "Scan result will be save at %s",
                    n_tasks, output_dir.resolve())
        logger.info(self.summarize())

        asyncio.run(self._scan(sequence,
                               executor=executor,
                               output_dir=output_dir,
                               chmod=chmod,
                               group=group))

        logger.info("Scan finished!")
