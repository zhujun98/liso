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

from .scan_param import JitterParam, SampleParam, ScanParam
from ..exceptions import LisoRuntimeError
from ..io import ExpWriter, SimWriter
from ..logging import logger


class _BaseScan(abc.ABC):
    def __init__(self):
        self._params = OrderedDict()
        self._param_dists = OrderedDict()

    def _check_param_name(self, name):
        return name

    def _add_scan_param(self, name, *, dist=-1., **kwargs):
        name = self._check_param_name(name)

        if name in self._params:
            raise ValueError(f"Parameter {name} already exists!")

        try:
            param = ScanParam(name, **kwargs)
        except TypeError:
            try:
                param = SampleParam(name, **kwargs)
            except TypeError:
                param = JitterParam(name, **kwargs)

        self._params[name] = param
        self._param_dists[name] = dist

    def _sort_param_sequence(self, seq):
        tol = tuple(self._param_dists.values())

        def _check_distance(a, b):
            for ia, ib, it in zip(a, b, tol):
                if abs(ia - ib) < it:
                    return False
            return True

        cache = list()
        ret_queue = deque()
        for item in zip(*seq):
            if not ret_queue:
                ret_queue.append(item)
            else:
                if _check_distance(item, ret_queue[-1]):
                    ret_queue.append(item)
                elif _check_distance(item, ret_queue[0]):
                    ret_queue.appendleft(item)
                else:
                    cache.append(item)
        n_pulses = len(cache) + len(ret_queue)

        for item in cache:
            length = len(ret_queue)
            for i in range(2, length-2, 2):
                if _check_distance(item, ret_queue[i-1]) and \
                        _check_distance(item, ret_queue[i]):
                    ret_queue.insert(i, item)
                    break

        if len(ret_queue) != n_pulses:
            return

        return list(ret_queue)

    def _generate_param_sequence(self, cycles):
        if not self._params:
            return []

        repeats = np.prod([len(param) for param in self._params.values()])
        ret = []
        for param in self._params.values():
            repeats = int(repeats / len(param))
            ret.append(param.generate(repeats=repeats, cycles=cycles))
            cycles *= len(param)

        for i in range(5):
            logger.debug(f"Generating scan parameter sequence (attempt {i+1})")
            seq = self._sort_param_sequence(ret)
            if seq is not None:
                return seq

        raise RuntimeError(
            "Failed to a parameter sequence with enough distance!")

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

    @abc.abstractmethod
    def _create_run_folder(self, parent):
        pass


class LinacScan(_BaseScan):
    """Class for performing scans in simulations."""
    def __init__(self, linac):
        """Initialization.

        :param Linac linac: Linac instance.
        """
        super().__init__()

        self._linac = linac

    def add_param(self, name, **kwargs):
        """Add a parameter for scan.

        :param str name: parameter name for simulations.
        """
        self._add_scan_param(name, **kwargs)

    def _check_param_name(self, name):
        """Override."""
        splitted = name.split('/', 1)
        first_bl = next(iter(self._linac))
        if len(splitted) == 1:
            return f"{first_bl}/{name}"
        return name

    def _create_run_folder(self, parent):
        parent_path = pathlib.Path(parent)
        # It is allowed to use an existing folder, but not an existing file.
        parent_path.mkdir(exist_ok=True)
        return parent_path

    async def _async_scan(self, cycles, run_folder, *,
                          start_id, n_tasks, group, chmod, **kwargs):
        tasks = set()
        sequence = self._generate_param_sequence(cycles)
        n_pulses = len(sequence)

        phasespace_schema = self._linac.schema
        control_schema = self._linac.compile(self._params)
        for param in control_schema:
            control_schema[param] = {'type': '<f4'}
        schema = (control_schema, phasespace_schema)

        with SimWriter(run_folder,
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

    def scan(self, cycles=1, folder='scan_data', *,
             start_id=1,
             n_tasks=None,
             chmod=True,
             group=1,
             seed=None,
             **kwargs):
        """Start a parameter scan.

        :param int cycles: number of cycles of the parameter space. For
            pure jitter study, it is the number of runs since the size
            of variable space is 1.
        :param str folder: folder where the simulation data will be saved.
        :param int start_id: starting simulation id. Default = 1.
        :param int/None n_tasks: maximum number of concurrent tasks.
        :param bool chmod: True for changing the permission to 400 after
            finishing writing.
        :param int group: writer group.
        :param int/None seed: seed for the legacy MT19937 BitGenerator
            in numpy.
        """
        if not isinstance(start_id, int) or start_id < 1:
            raise ValueError(
                f"start_id must a positive integer. Actual: {start_id}")

        if n_tasks is None:
            n_tasks = multiprocessing.cpu_count()

        run_folder = self._create_run_folder(folder)

        logger.info(str(self._linac))
        logger.info(f"Starting parameter scan with {n_tasks} CPUs.")
        logger.info(self.summarize())

        np.random.seed(seed)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._async_scan(
            cycles, run_folder,
            start_id=start_id,
            n_tasks=n_tasks,
            group=group,
            chmod=chmod,
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

        self._param_readouts = dict()

    def _create_run_folder(self, parent):
        parent_path = pathlib.Path(parent)
        # It is allowed to use an existing parent folder, but not a run folder.
        parent_path.mkdir(exist_ok=True)

        next_run = 1  # starting from 1
        for d in parent_path.iterdir():
            # Here d could also be a file
            if re.search(r'r\d{4}', d.name):
                seq = int(d.name[1:])
                if seq >= next_run:
                    next_run = seq + 1

        next_run_folder = parent_path.joinpath(f'r{next_run:04d}')
        next_run_folder.mkdir(parents=True, exist_ok=False)
        return next_run_folder

    def scan(self, cycles=1, folder='scan_data', *,
             tasks=None,
             chmod=True,
             timeout=None,
             group=1,
             seed=None):
        """Start a parameter scan.

        :param int cycles: number of cycles of the parameter space. For
            pure jitter study, it is the number of runs since the size
            of variable space is 1.
        :param str folder: folder in which data for each run will be stored in
            in its own sub-folder.
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

        run_folder = self._create_run_folder(folder)

        sequence = self._generate_param_sequence(cycles)
        n_pulses = len(sequence) if sequence else cycles
        with ExpWriter(run_folder,
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

        :param str name: the DOOCS address.
        :param str/None readout: the DOOCS address for validating the value
            being written. If None, the written value will not be validated.
        :param float tol: tolerance for the validation. Positive value for
            absolute error and negative value for relative error.
        """
        self._add_scan_param(name, **kwargs)

        self._param_readouts[name] = {'readout': readout}

        self._param_readouts[name]['tol'] = tol
