"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import sys
import traceback

import numpy as np

from .scan_param import JitterParam, SampleParam, ScanParam
from ..exceptions import LisoRuntimeError
from ..io import ExpWriter, SimWriter
from ..logging import logger


class _BaseScan:
    def __init__(self):
        self._params = OrderedDict()

    def add_param(self, name, *args, **kwargs):
        """Add a parameter for scan.

        :param str name: parameter name for simulations and a valid
            DOOCS address for experiments.
        """
        # TODO: check format of address

        if name in self._params:
            raise ValueError(f"Parameter {name} already exists!")

        try:
            param = ScanParam(name, *args, **kwargs)
        except TypeError:
            try:
                param = SampleParam(name, *args, **kwargs)
            except TypeError:
                param = JitterParam(name, *args, **kwargs)

        self._params[name] = param

    def _generate_param_sequence(self, cycles, seed):
        np.random.seed(seed)  # set the random state!
        repeats = np.prod([len(param) for param in self._params.values()])
        ret = []
        for param in self._params.values():
            repeats = int(repeats / len(param))
            ret.append(param.generate(repeats=repeats, cycles=cycles))
            cycles *= len(param)

        return list(zip(*ret))

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
            if isinstance(param, ScanParam):
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


class LinacScan(_BaseScan):
    """Class for performing scans in simulations."""
    def __init__(self, linac):
        """Initialization.

        :param Linac linac: Linac instance.
        """
        super().__init__()

        self._linac = linac

    async def _async_scan(self, n_tasks, output, *,
                          cycles, n_particles, start_id, seed, **kwargs):
        tasks = set()
        sequence = self._generate_param_sequence(cycles, seed)
        n_pulses = len(sequence)

        with SimWriter(n_pulses, n_particles, output, start_id=start_id) \
                as writer:
            count = 0
            while True:
                if count < n_pulses:
                    x_map = dict()
                    for i, k in enumerate(self._params):
                        x_map[k] = sequence[count][i]

                    sim_id = count + start_id
                    task = asyncio.ensure_future(
                        self._linac.async_run(
                            count, x_map, f'tmp{sim_id:06d}', **kwargs))
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
                            idx, controls, phasespaces = task.result()
                            writer.write(idx, controls, phasespaces)
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

    def scan(self,
             n_tasks=None, *,
             cycles=1,
             n_particles=2000,
             output='scan.hdf5',
             start_id=1,
             seed=None,
             **kwargs):
        """Start a parameter scan.

        :param int/None n_tasks: maximum number of concurrent tasks.
        :param int cycles: number of cycles of the parameter space. For
            pure jitter study, it is the number of runs since the size
            of variable space is 1.
        :param int n_particles: number of particles to be stored.
        :param str output: output file.
        :param int start_id: starting simulation id. Default = 1.
        :param int/None seed: seed for the legacy MT19937 BitGenerator
            in numpy.
        """
        if n_tasks is None:
            n_tasks = multiprocessing.cpu_count()

        logger.info(str(self._linac))
        logger.info(f"Starting parameter scan with {n_tasks} CPUs.")
        logger.info(self.summarize())

        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._async_scan(
            n_tasks, output,
            cycles=cycles,
            n_particles=n_particles,
            start_id=start_id,
            seed=seed,
            **kwargs))

        logger.info(f"Scan finished!")


class MachineScan(_BaseScan):
    """Class for performing scans with a real machine."""
    def __init__(self, machine):
        """Initialization.

        :param _BaseMachine machine: Machine instance.
        """
        super().__init__()

        self._machine = machine

    def scan(self, *,
             cycles=1,
             output='scan.hdf5',
             n_tasks=None,
             seed=None):
        """Start a parameter scan.

        :param int cycles: number of cycles of the parameter space. For
            pure jitter study, it is the number of runs since the size
            of variable space is 1.
        :param str output: output file.
        :param int/None n_tasks: maximum number of concurrent tasks for
            read and write.
        :param int/None seed: seed for the legacy MT19937 BitGenerator
            in numpy.
        """
        if n_tasks is None:
            n_tasks = multiprocessing.cpu_count()

        logger.info(f"Starting parameter scan with {n_tasks} CPUs.")
        logger.info(self.summarize())

        executor = ThreadPoolExecutor(max_workers=n_tasks)

        sequence = self._generate_param_sequence(cycles, seed)
        n_pulses = len(sequence) if sequence else cycles
        with ExpWriter(output, schema=self._machine.schema) as writer:
            count = 0
            while count < n_pulses:
                mapping = dict()
                if sequence:
                    for i, k in enumerate(self._params):
                        mapping[k] = sequence[count][i]

                count += 1
                logger.info(f"Scan {count:06d}: "
                            + str(mapping)[1:-1].replace(': ', ' = '))

                try:
                    idx, controls, instruments = self._machine.run(
                        executor=executor, mapping=mapping)
                    writer.write(idx, controls, instruments)
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
