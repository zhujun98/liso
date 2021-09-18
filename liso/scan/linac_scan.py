"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import asyncio
import multiprocessing
import pathlib
import sys
from typing import Optional
import traceback

import numpy as np

from .base_scan import BaseScan
from ..exceptions import LisoRuntimeError
from ..io import SimWriter
from ..simulation import Linac
from ..logging import logger


class LinacScan(BaseScan):
    """Class for performing scans in simulations."""
    def __init__(self, linac: Linac) -> None:
        """Initialization.

        :param linac: Linac instance.
        """
        super().__init__()

        self._linac = linac

    def _check_param_name(self, name: str):
        """Override."""
        splitted = name.split('/', 1)
        first_bl = next(iter(self._linac))
        if len(splitted) == 1:
            return f"{first_bl}/{name}"
        return name

    def _create_output_dir(self, parent):
        parent_path = pathlib.Path(parent)
        # It is allowed to use an existing directory, but not an existing file.
        parent_path.mkdir(exist_ok=True)
        return parent_path

    async def _async_scan(self, cycles, output_dir, *,
                          start_id, n_tasks, group, chmod, **kwargs):
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

                    logger.info(f"Scan {sim_id:06d}: "
                                + str(x_map)[1:-1].replace(': ', ' = '))

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

                        tasks.remove(task)

    def scan(self, cycles: int = 1, output_dir: str = "./", *,
             start_id: int = 1,
             n_tasks: Optional[int] = None,
             chmod: bool = True,
             group: int = 1,
             seed: Optional[int] = None,
             **kwargs) -> None:
        """Start a parameter scan.

        :param cycles: Number of cycles of the parameter space. For
            pure jitter study, it is the number of runs since the size
            of variable space is 1.
        :param output_dir: Directory where the output simulation data is saved.
        :param start_id: Starting simulation id. Default = 1.
        :param n_tasks: Maximum number of concurrent tasks.
        :param chmod: True for changing the permission to 400 after
            finishing writing.
        :param group: Writer group.
        :param seed: Seed for the legacy MT19937 BitGenerator in numpy.
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
        logger.info(f"Starting parameter scan with {n_tasks} CPUs.")
        logger.info(self.summarize())

        np.random.seed(seed)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._async_scan(
            cycles, output_dir,
            start_id=start_id,
            n_tasks=n_tasks,
            group=group,
            chmod=chmod,
            **kwargs))

        logger.info(f"Scan finished!")
