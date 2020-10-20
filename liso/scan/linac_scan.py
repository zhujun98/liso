"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import asyncio
from collections import OrderedDict
import functools
import multiprocessing
import sys
import traceback
from threading import Thread

import numpy as np

from .scan_param import JitterParam, SampleParam, ScanParam
from ..io import SimWriter
from ..logging import logger


def run_in_thread(daemon=False):
    """Run a function/method in a thread."""
    def wrap(f):
        @functools.wraps(f)
        def threaded_f(*args, **kwargs):
            t = Thread(target=f, daemon=daemon, args=args, kwargs=kwargs)
            t.start()
            return t
        return threaded_f
    return wrap


class LinacScan(object):
    def __init__(self, linac, *, name=''):
        """Initialization.

        :param Linac linac: Linac instance.
        :param str name: Name of the parameter_scan problem.
        """
        self.name = name

        self._linac = linac

        self._params = OrderedDict()

    def add_param(self, name, *args, **kwargs):
        """Add a parameter for scan.

        :param str name: parameter name.
        """
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

    async def _async_scan(self, n_tasks, output, *,
                          cycles, n_particles, start_id, seed, **kwargs):
        tasks = set()
        sequence = self._generate_param_sequence(cycles, seed)
        n_pulses = len(sequence)
        writer = SimWriter(n_pulses, n_particles, output, start_id=start_id)
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
                    except RuntimeError as e:
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

    def summarize(self):
        text = '\n' + '=' * 80 + '\n'
        text += 'Scanned parameters: %s\n' % self.name
        text += self.__str__()
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

    def __str__(self):
        text = ''
        return text
