#!/usr/bin/python
"""
Author: Jun Zhu
"""
from collections import OrderedDict

import numpy as np

from ..operation import Operation
from .jitter import Jitter
from .response import Response

from ..simulation.simulation_utils import check_templates


class LinacJitter(Operation):
    """Inherited from LinacOperation."""
    def __init__(self, linac, *, name='unnamed'):
        """Initialization."""
        super().__init__(name)

        self._linac = linac

        self.jitters = OrderedDict()
        self.responses = OrderedDict()
        self._x_map = dict()

        self._outcome = False  # a flag used to control __str__ output

    def add_jitter(self, name, **kwargs):
        """Add a jitter."""
        try:
            self.jitters[name] = Jitter(name, **kwargs)
            self._x_map[name] = self.jitters[name].value
        except (TypeError, ValueError):
            print("Input is not valid for a Jitter class instance!")
            raise

    def del_jitter(self, name):
        """Delete a jitter by name."""
        try:
            del self.jitters[name]
            del self._x_map[name]
        except KeyError:
            raise KeyError("{} is not a jitter!".format(name))

    def add_response(self, name, **kwargs):
        """Add a response."""
        try:
            self.responses[name] = Response(name, **kwargs)
        except (TypeError, ValueError):
            print("Input is not valid for a Response class instance!")
            raise

    def del_response(self, name):
        """Delete a response by name."""
        try:
            del self.responses[name]
        except KeyError:
            raise KeyError("{} is not a response!".format(name))

    def run(self, passes):
        """Run the simulations.

        :param passes: int
            Number of independent runs.
        """
        check_templates(self._linac.get_templates(), self._x_map)

        self._outcome = False
        print(self.__str__())

        # Generate random numbers for all passes
        random_numbers = self._generate_randoms(len(self.jitters), passes)

        for i in range(passes):
            self._update_x_map(random_numbers[i, :])
            self._linac.update(self._x_map)
            self._update_response(i+1)

        self._outcome = True
        print(self.__str__())

    @staticmethod
    def _generate_randoms(n, size):
        """Generate multi-dimensional random numbers.

        Each dimension has a mean of 0 and a standard deviation of 1.0.

        :param n: int
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

    def _update_x_map(self, x):
        """Update values in x_map.

        :param x: 1D array like
            Normalized (mean = 0, std = 1.0) random numbers.
        """
        for key, v in zip(self._x_map.keys(), x):
            self._x_map[key] = self.jitters[key].value + v*self.jitters[key].sigma

    def _update_response(self, count):
        """Update all Responses.

        :param count: int
            Count of simulations.
        """
        for item in self.responses.values():
            if item.func is not None:
                item.values.append(item.func(self._linac))
            else:
                keys = item.expr
                if keys[1] in ('max', 'min', 'start', 'end', 'ave', 'std'):
                    item.values.append(self._linac.__getattr__(keys[0]).__getattribute__(
                        keys[1]).__getattribute__(keys[2]) * item.scale)
                else:
                    item.values.append(self._linac.__getattr__(keys[0]).__getattr__(
                        keys[1]).__getattribute__(keys[2]) * item.scale)

        if self.printout > 0:
            out = []
            for item in self.responses.values():
                out.append(item.values[-1])
            print('{:04d} - '.format(count),
                  ['{:9.6e}'.format(v) for v in out])

    def __str__(self):
        if self._outcome is False:
            text = '\nJitter problem: %s' % self.name
        else:
            text = '\nOutcome for jitter problem: %s' % self.name

        text += '\n' + '='*80 + '\n'
        text += self._format_item(self.responses, 'Response(s)')
        text += self._format_item(self.jitters, 'Jitter(s)')
        return text

    @staticmethod
    def _format_item(instance_set, name):
        """Return structured list of parameters.

        :param instance_set: OrderedDict
            Should be either self.jitters, self.covariables or self.responses.
        :param name: string
            Name of the instance set.
        """
        is_first = True
        text = ''
        for ele in instance_set.values():
            if is_first is True:
                text += '\n' + name + ':\n'
                text += str(ele)
                is_first = False
            else:
                text += ele.list_item()

        return text
