"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import asyncio
import multiprocessing
from pathlib import Path
import sys
from typing import Optional
import traceback

import numpy as np

from .abstract_scan import AbstractScan
from ..exceptions import LisoRuntimeError
from ..io import SimWriter
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

    @staticmethod
    def _create_output_dir(parent: str) -> Path:
        """Maybe create a directory to store the output data."""
        parent_path = Path(parent)
        # It is allowed to use an existing directory, but not an existing file.
        parent_path.mkdir(exist_ok=True)
        return parent_path

    async def _scan_imp(self, cycles: int,  # pylint: disable=too-many-locals
                        output_dir: Path, *,
                        start_id: int,
                        n_tasks: int,
                        group: int,
                        chmod: bool,
                        **kwargs) -> None:
        tasks = set()
        sequence = self._generate_param_sequence(cycles)
        n_pulses = len(sequence)

        phasespace_schema = self._linac.schema
        control_schema = self._linac.compile(self._params)
        for param in control_schema:
            control_schema[param] = {'type': '<f4'}
        schema = (control_schema, phasespace_schema)

        with SimWriter(output_dir,
                       schema=schema,
                       chmod=chmod,
                       group=group) as writer:
            count = 0
            while True:
                if count < n_pulses:
                    x_map = dict()
                    for i, k in enumerate(self._params):
                        x_map[k] = sequence[count][i]

                    sim_id = count + start_id
                    task = asyncio.create_task(
                        self._linac.async_run(sim_id, x_map, **kwargs))
                    tasks.add(task)

                    logger.info("Scan %06d: %s",
                                sim_id, str(x_map)[1:-1].replace(': ', ' = '))

                    count += 1

                if len(tasks) == 0:
                    break

                if len(tasks) >= n_tasks or count == n_pulses:
                    done, _ = await asyncio.wait(
                        tasks, return_when=asyncio.FIRST_COMPLETED)

                    for task in done:
                        try:
                            sim_id, controls, phasespaces = task.result()
                            writer.write(sim_id, controls, phasespaces)
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

                        tasks.remove(task)

    def scan(self, cycles: int = 1, *,
             start_id: int = 1,
             n_tasks: Optional[int] = None,
             chmod: bool = True,
             group: int = 1,
             seed: Optional[int] = None,
             output_dir: str = "./",
             **kwargs) -> None:
        """Run the scan.

        :param cycles: Number of cycles of the parameter space. For
            pure jitter study, it is the number of runs since the size
            of variable space is 1.
        :param start_id: Starting simulation id. Default = 1.
        :param n_tasks: Maximum number of concurrent tasks.
        :param chmod: True for changing the permission to 400 after
            finishing writing.
        :param group: Writer group.
        :param seed: Seed for the legacy MT19937 BitGenerator in numpy.
        :param output_dir: Directory where the output simulation data is saved.
        """
        if not self._params:
            raise ValueError("No scan parameters specified!")

        if not isinstance(start_id, int) or start_id < 1:
            raise ValueError(
                f"start_id must a positive integer. Actual: {start_id}")

        if n_tasks is None:
            n_tasks = multiprocessing.cpu_count()

        output_dir = self._create_output_dir(output_dir)

        logger.info(str(self._linac))
        logger.info("Starting parameter scan with %s CPUs.", n_tasks)
        logger.info(self.summarize())

        np.random.seed(seed)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._scan_imp(
            cycles, output_dir,
            start_id=start_id,
            n_tasks=n_tasks,
            group=group,
            chmod=chmod,
            **kwargs))

        logger.info("Scan finished!")
