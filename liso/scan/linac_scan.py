"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import asyncio
import multiprocessing
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .abstract_scan import AbstractScan
from ..io import create_next_run_folder, SimWriter
from ..simulation import Linac
from ..logging import logger


class LinacScan(AbstractScan):
    """Class for performing scans in simulations."""
    def __init__(self, linac: Linac) -> None:
        """Initialization.

        :param linac: Linac instance.
        """
        super().__init__()

        self._linac = linac

    def _parse_param_name(self, name: str) -> str:
        """Override."""
        splitted = name.split('/', 1)
        first_bl = next(iter(self._linac))
        if len(splitted) == 1:
            return f"{first_bl}/{name}"
        return name

    def _create_schema(self) -> dict:
        phasespace_schema = self._linac.schema
        control_schema = self._linac.compile(self._params)
        for param in control_schema:
            control_schema[param] = {'type': '<f4'}
        return {
            "control": control_schema,
            "phasespace": phasespace_schema
        }

    async def _scan_imp(self, sequence: list,
                        writer: SimWriter, *,
                        start_id: int,
                        n_tasks: int,
                        timeout: Optional[int] = None) -> None:
        tasks = set()
        n_pulses = len(sequence)

        self._linac.check_temp_swd(start_id, start_id + n_pulses)

        count = 0
        while True:
            if count < n_pulses:
                mapping = dict()
                for i, k in enumerate(self._params):
                    mapping[k] = sequence[count][i]
                sim_id = count + start_id
                count += 1
                logger.info("Scan %06d: %s",
                            sim_id, str(mapping)[1:-1].replace(': ', ' = '))

                task = asyncio.create_task(
                    self._linac.async_run(sim_id, mapping, timeout=timeout))
                tasks.add(task)

            if len(tasks) == 0:
                break

            if len(tasks) >= n_tasks or count == n_pulses:
                done, _ = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED)

                for task in done:
                    self._collect_result(writer, task.result)
                    tasks.remove(task)

    def scan(self, cycles: int = 1, *,  # pylint: disable=arguments-differ
             n_tasks: Optional[int] = None,
             output_dir: Union[str, Path] = "./",
             chmod: bool = True,
             group: int = 1,
             seed: Optional[int] = None,
             start_id: int = 1,
             timeout: Optional[int] = None) -> None:
        """Run the scan.

        :param cycles: Number of cycles of the parameter space. For
            pure jitter study, it is the number of runs since the size
            of variable space is 1.
        :param n_tasks: Maximum number of concurrent tasks.
        :param output_dir: Directory where the output simulation data is saved.
        :param chmod: True for changing the permission to 400 after
            finishing writing.
        :param group: Writer group.
        :param seed: Seed for the legacy MT19937 BitGenerator in numpy.
        :param start_id: Starting simulation id. Default = 1.
        :param timeout: Timeout in seconds for running a single simulation.
            None for no timeout.

        :raises ValueError: If generation of parameter sequence fails.
        :raises FileExistsError: If there is already any directory which has
            the same name as the temporary simulation directory to be created.
        :raises LisoRuntimeError: If any Beamline of the Linac cannot create
            a temporary directory to run simulation.
        """
        if not self._params:
            raise ValueError("No scan parameters specified!")

        if not isinstance(start_id, int) or start_id < 1:
            raise ValueError(
                f"start_id must a positive integer. Actual: {start_id}")

        if n_tasks is None:
            n_tasks = multiprocessing.cpu_count()

        output_dir = create_next_run_folder(output_dir, sim=True)

        schema = self._create_schema()

        sequence = self._generate_param_sequence(cycles)

        logger.info(str(self._linac))
        logger.info("Starting parameter scan with %s CPUs. "
                    "Scan result will be save at %s",
                    n_tasks, output_dir.resolve())
        logger.info(self.summarize())

        np.random.seed(seed)
        loop = asyncio.get_event_loop()
        with SimWriter(output_dir,
                       schema=schema,
                       chmod=chmod,
                       group=group) as writer:
            loop.run_until_complete(self._scan_imp(
                sequence, writer,
                start_id=start_id,
                n_tasks=n_tasks,
                timeout=timeout))

        logger.info("Scan finished!")
