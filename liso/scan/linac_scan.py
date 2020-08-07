"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import asyncio
from collections import OrderedDict
import functools
from threading import Thread

import numpy as np

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
        self._n = None

        self._x_map = dict()

    def add_param(self, name, values):
        """Add a parameter for scan.

        :param str name: parameter name.
        :param array-like values: a list of values for scanning.
        """
        if self._n is None:
            self._n = len(values)
        else:
            if len(values) != self._n:
                raise ValueError(f"Parameters have different lengths: "
                                 f"{self._n} and {len(values)}")

        if name in self._params:
            logger.warning(f"Overwrite existing parameter: {name}!")

        self._params[name] = values

    async def _async_scan(self, n_tasks, *args, **kwargs):
        tasks = set()
        count = 0
        while True:
            if count < self._n:
                for k, v in self._params.items():
                    self._x_map[k] = v[count]

                logger.info(f"Simulation {count+1:07d}: "
                            + str(self._x_map)[1:-1].replace(': ', ' = '))

                task = asyncio.ensure_future(
                    self._linac.async_run(self._x_map, *args, **kwargs))
                tasks.add(task)

                count += 1

            if len(tasks) == 0:
                break

            if len(tasks) >= n_tasks or count == self._n:
                done, _ = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED)

                for task in done:
                    tasks.remove(task)

    def scan(self, n_tasks=1, *args, **kwargs):
        """Start a parameter scan.

        :param int n_tasks: maximum number of concurrent tasks.
        """
        logger.info(str(self._linac) + self._get_info())

        asyncio.run(self._async_scan(n_tasks, *args, **kwargs))

        logger.info(f"Scan finished!")

    @staticmethod
    def _generate_randoms(n, size):
        """Generate multi-dimensional random numbers.

        Each dimension has a mean of 0 and a standard deviation of 1.0.

        :param int n: int
            Dimensions.
        :param size: int
            Length.

        :return: numpy.ndarray
            A n by size 2D array
        """
        mean = np.zeros(n)
        cov = np.zeros([n, n])
        np.fill_diagonal(cov, 1.0)

        return np.random.multivariate_normal(mean, cov, size)

    def _get_info(self):
        text = '\n' + '=' * 80 + '\n'
        text += 'Parameter scan: %s\n' % self.name
        text += self.__str__()
        text += '\n' + '=' * 80 + '\n'
        return text

    def __str__(self):
        text = ''
        return text