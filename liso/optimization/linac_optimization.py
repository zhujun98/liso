#!/usr/bin/python

"""
The optimization problem is define as:

min f(x) w.r.t. x

s.t. g_j(x) = 0, j = 1, ..., m_e

    g_j(x) <= 0, j = m_e + 1, ..., m

    x_i_L <= x_i <= x_i_U, i = 1, ..., n

where:

    x is the vector of design variables;
    f(x) is a nonlinear function;
    g(x) is a linear or nonlinear function;
    n is the number of design variables;
    m_e is the number of equality constraints;
    m is the total number of constraints (number of equality
    constraints: m_i = m - m_e).


Author: Jun Zhu

"""
from collections import OrderedDict
from itertools import chain

import numpy as np

from ..simulation import Linac
from .variable import Variable
from .covariable import Covariable
from .constraint import EConstraint
from .constraint import IConstraint
from .objective import Objective
from ..config import Config
from ..exceptions import *

INF = Config.INF


class LinacOptimization(object):
    """LinacOpt class."""
    def __init__(self, linac, *,
                 name='',
                 max_successive_failures=20,
                 workers=1):
        """Initialization.

        :param linac: Linac object
            Linac instance.
        :param name: str
            Name of the optimization problem (arbitrary).
        :param max_successive_failures: int
            Max number of allowed successive failures.
        :param workers: int
            Number of threads.
        """
        if isinstance(linac, Linac):
            self._linac = linac
        else:
            raise TypeError("{} is not a Linac instance!".format(linac))

        self.name = name

        self.variables = OrderedDict()
        self.covariables = OrderedDict()
        self.objectives = OrderedDict()
        self.e_constraints = OrderedDict()
        self.i_constraints = OrderedDict()

        self._x_map = OrderedDict()  # Initialize the x_map dictionary

        self._workers = 1
        self.workers = workers

        self._nf = 0
        self._max_successive_failures = max_successive_failures

        self._DEBUG = False

    @property
    def workers(self):
        return self._workers

    @workers.setter
    def workers(self, value):
        if isinstance(value, int) and value > 0:
            self._workers = value

    def solve(self, optimizer):
        """Run the optimization and print the result.

        :param optimizer: Optimizer object.
            Optimizer.
        """
        print(self.__str__())
        opt_f, opt_x, _ = optimizer(self)
        self._create_solution(opt_x)
        self._verify_solution(opt_f)
        print(self.__str__())

    def eval_obj_cons(self, x):
        """Objective-constraint function.

        This method will be called in the f_obj_con function of the optimizer.

        :param x: array-like
            Variables updated after each iteration.
        """
        # Run simulations with the new input files
        is_update_failed = True
        try:
            self._update_x_map(x)
            self._linac.update(self._x_map, self.workers)
            is_update_failed = False
            self._nf = 0
        # exception propagates from Beamline.simulate() method
        except SimulationNotFinishedProperlyError as e:
            print(e)
            self._nf += 1
        # exception propagates from BeamParameters.update() method
        except FileNotFoundError as e:
            print(e)
            self._nf += 1
        except Exception as e:
            print("Unknown bug detected!")
            raise
        finally:
            if self._nf > self._max_successive_failures:
                raise SimulationSuccessiveFailureError(
                    "Maximum allowed number of successive failures reached!")

        if is_update_failed is False:
            self._update_obj_cons()

            f = []
            for obj in self.objectives.values():
                f.append(obj.value)
            g = []
            for con in chain(self.e_constraints.values(),
                             self.i_constraints.values()):
                g.append(con.value)

            self._nf = 0
        else:
            f = [INF] * len(self.objectives)
            g = [INF] * (len(self.i_constraints) + len(self.e_constraints))

        if self._DEBUG is True:
            print(['{:^12}: {:9.2e}'.format(key, value) for key, value in self._x_map.items()],
                  ['{:9.2e}'.format(v) for v in f],
                  ['{:9.2e}'.format(v) for v in g],
                  is_update_failed)

        return f, g, is_update_failed

    def _create_solution(self, opt_x):
        """Update attributes with the optimized values.

        Update objectives, variables, covariables, constraints.

        :param opt_x: list
            Optimized x values.
        """
        self._update_x_map(opt_x)
        self._linac.update(self._x_map, self.workers)
        self._update_obj_cons()

    def _update_x_map(self, x):
        """Update values in x_map."""
        x_covar = [0] * len(self.covariables)  # placeholder
        for key, v in zip(self._x_map.keys(), chain(x, x_covar)):
            self._x_map[key] = v
            try:
                self.variables[key].value = v
            except KeyError:
                var = self.covariables[key].dependent
                a = self.covariables[key].scale
                b = self.covariables[key].shift
                self._x_map[key] = a * self._x_map[var] + b

    def _update_obj_cons(self):
        """Update all Objective and Constraints."""
        for item in chain(self.objectives.values(),
                          self.e_constraints.values(),
                          self.i_constraints.values()):
            if item.func is not None:
                item.value = item.func(self._linac)
            else:
                keys = item.expr
                if keys[1] in ('max', 'min', 'start', 'end', 'ave', 'std'):
                    item.value = self._linac.__getattr__(keys[0]).__getattribute__(
                        keys[1]).__getattribute__(keys[2])
                else:
                    item.value = self._linac.__getattr__(keys[0]).__getattr__(
                        keys[1]).__getattribute__(keys[2])

    def add_var(self, name, **kwargs):
        """Add a variable."""
        try:
            self.variables[name] = Variable(name, **kwargs)
            self._x_map[name] = self.variables[name].value
        except (TypeError, ValueError):
            print("Input is not valid for a Variable class instance!")
            raise

    def del_var(self, name):
        """Delete a variable by name."""
        try:
            del self.variables[name]
            del self._x_map[name]
        except KeyError:
            raise KeyError("{} is not a variable!".format(name))

    def add_covar(self, name, *args, **kwargs):
        """Add a covariable."""
        try:
            self.covariables[name] = Covariable(name, *args, **kwargs)

            var = self.covariables[name].dependent
            a = self.covariables[name].scale
            b = self.covariables[name].shift
            try:
                self._x_map[name] = a * self._x_map[var] + b
            except KeyError:
                raise ValueError("The dependent variable '%s' does not exist!" % var)
        except (TypeError, ValueError):
            print("Input is not valid for a Covariable class instance!")
            raise

    def del_covar(self, name):
        """Delete a covariable by name."""
        try:
            del self.covariables[name]
            del self._x_map[name]
        except KeyError:
            raise KeyError("{} is not a covariable!".format(name))

    def add_obj(self, name, *args, **kwargs):
        """Add an objective."""
        try:
            self.objectives[name] = Objective(name, *args, **kwargs)
        except (TypeError, ValueError):
            print("Input is not valid for an Objective class instance!")
            raise

    def del_obj(self, name):
        """Delete an objective by name"""
        try:
            del self.objectives[name]
        except KeyError:
            raise KeyError("{} is not an objective!".format(name))

    def add_icon(self, name, *args, **kwargs):
        """Add an inequality constraint."""
        try:
            self.i_constraints[name] = IConstraint(name, *args, **kwargs)
        except (TypeError, ValueError):
            print("Input is not valid for an IConstraint class instance!")
            raise

    def del_icon(self, name):
        """Delete an inequality constraint by name."""
        try:
            del self.i_constraints[name]
        except KeyError:
            raise KeyError("{} is not an inequality constraint!".format(name))

    def add_econ(self, name, *args, **kwargs):
        """Add an equality constraint."""
        try:
            self.e_constraints[name] = EConstraint(name, *args, **kwargs)
        except (TypeError, ValueError):
            print("Input is not valid for an EConstraint class instance!")
            raise

    def del_econ(self, name):
        """Delete an equality constraint by name."""
        try:
            del self.e_constraints[name]
        except KeyError:
            raise KeyError("{} is not an equality constraint!".format(name))

    def __str__(self):
        text = '\nOptimization Problem: %s' % self.name
        text += '\n' + '='*80 + '\n'
        text += self._format_item(self.objectives, 'Objective(s)')
        text += self._format_item(self.e_constraints, 'Equality constraint(s)')
        text += self._format_item(self.i_constraints, 'Inequality constraint(s)')
        text += self._format_item(self.variables, 'Variable(s)')
        text += self._format_item(self.covariables, 'Covariable(s)')
        return text

    @staticmethod
    def _format_item(instance_set, title):
        """Return structured list of parameters.

        :param instance_set: OrderedDict
            Should be either self.objectives, self.constraints or
            self.variables.
        :param title: string
            Title of the output.
        """
        is_first = True
        text = ''
        for ele in instance_set.values():
            if is_first is True:
                text += '\n' + title + ':\n'
                text += str(ele)
                is_first = False
            else:
                text += repr(ele)
        return text

    def _verify_solution(self, opt_f):
        """Verify the solution.

        :param opt_f: list
            Optimized objective value(s).
        """
        if self._DEBUG is True:
            f = [item.value for item in self.objectives.values()]
            np.testing.assert_almost_equal(f, opt_f)
