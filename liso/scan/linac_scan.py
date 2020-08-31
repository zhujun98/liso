"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import asyncio
from collections import OrderedDict
import functools
import itertools
import sys
import traceback
from threading import Thread

from .scan_param import ScanParam
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
    def __init__(self, linac, *, name='scan_prob'):
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

        self._params[name] = ScanParam(name, *args, **kwargs)

    def _generate_param_sequence(self, repeat):
        ret = []
        for i in range(repeat):
            for param in self._params.values():
                param.reset()
            ret.extend(itertools.product(*self._params.values()))
        return ret

    async def _async_scan(self, n_tasks, output, repeat=1, **kwargs):
        x_map = dict()

        tasks = set()
        sequence = self._generate_param_sequence(repeat)
        num = len(sequence)
        writer = SimWriter(num, output)
        count = 0
        while True:
            if count < num:
                for i, k in enumerate(self._params):
                    x_map[k] = sequence[count][i]

                task = asyncio.ensure_future(
                    self._linac.async_run(
                        count, x_map, f'tmp{count:06d}', **kwargs))
                tasks.add(task)

                count += 1

                logger.info(f"Scan {count:06d}: "
                            + str(x_map)[1:-1].replace(': ', ' = '))

            if len(tasks) == 0:
                break

            if len(tasks) >= n_tasks or count == num:
                done, _ = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED)

                for task in done:
                    try:
                        idx, output = task.result()
                        writer.write(idx, output)
                    except RuntimeError as e:
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        logger.debug(repr(traceback.format_tb(exc_traceback))
                                     + str(e))
                        logger.warning(f"Scan {count:06d}: " + str(e))
                    except Exception as e:
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        logger.error(
                            f"Scan {count:06d} (Unexpected exceptions): "
                            + repr(traceback.format_tb(exc_traceback))
                            + str(e))
                        raise

                    tasks.remove(task)

    def scan(self, n_tasks=1, *, repeat=1, output='scan.hdf5', **kwargs):
        """Start a parameter scan.

        :param int n_tasks: maximum number of concurrent tasks.
        :param int repeat: number of repeats of the parameter space. For
            pure jitter study, it is the number of runs since the size
            of variable space is 1.
        :param str output: output file.
        """
        logger.info(str(self._linac) + self._get_info())

        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            self._async_scan(n_tasks, output, repeat=repeat, **kwargs))

        logger.info(f"Scan finished!")

    def _get_info(self):
        text = '\n' + '=' * 80 + '\n'
        text += 'Parameter scan: %s\n' % self.name
        text += self.__str__()
        text += '\n' + '=' * 80 + '\n'
        return text

    def __str__(self):
        text = ''
        return text
