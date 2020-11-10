"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""

# The optimization problem is define as:
#
# min f(x) w.r.t. x
#
# s.t. g_j(x) = 0, j = 1, ..., m_e
#
#     g_j(x) <= 0, j = m_e + 1, ..., m
#
#     x_i_L <= x_i <= x_i_U, i = 1, ..., n
#
# where:
#
#     x is the vector of design variables;
#     f(x) is a nonlinear function;
#     g(x) is a linear or nonlinear function;
#     n is the number of design variables;
#     m_e is the number of equality constraints;
#     m is the total number of constraints (number of equality
#     constraints: m_i = m - m_e).

from collections import OrderedDict
from itertools import chain
import math
import sys
import time
import traceback

import numpy as np

from .variable import Variable
from ..covariable import Covariable
from .constraint import EConstraint
from .constraint import IConstraint
from .objective import Objective
from ..exceptions import LisoRuntimeError
from ..logging import logger, opt_logger


class Optimization(object):
    """Inherited from Operation.

    Attributes:
        variables (OrderedDict): variable set.
        covariables (OrderedDict): covariable set.
        objectives (OrderedDict): objective set.
        e_constraints (OrderedDict): equality constraint set.
        i_constraints (OrderedDict): inequality constraint set.
    """
    def __init__(self, name='opt_prob', *, opt_func=None):
        """Initialization.

        :param str name: Name of the optimization problem.
        :param callable opt_func: A callable object which returns (objective,
                                  constraints).
        """
        self.name = name

        self.variables = OrderedDict()
        self.covariables = OrderedDict()
        self.objectives = OrderedDict()
        self.e_constraints = OrderedDict()
        self.i_constraints = OrderedDict()

        self._x_map = OrderedDict()  # Initialize the x_map dictionary

        self._nfeval = 0  # No. of evaluations

        self._opt_func = opt_func

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
                raise ValueError(
                    "The dependent variable '%s' does not exist!" % var)
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

    def _update_x_map(self, x):
        """Update values in x_map.

        Invoked in self.eval_objs_cons(). Not raise.

        :param x: 1D array like
            New variable values.
        """
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

    def eval_objs_cons(self, x, *args, **kwargs):
        """Objective-constraint function.

        This method will be called in the f_obj_con function defined within
        optimizer.__call__().

        :param x: array-like
            Variables updated after each iteration.

        :return: f: list
            Evaluations of objective functions.
        :return: g: list
            Evaluations of constraint functions.
        :return: is_update_failed: bool
            An indicator which indicates whether the function evaluation fails.
        """
        self._update_x_map(x)

        x = list(self._x_map.values())
        f_eval, g_eval = self._opt_func(x)
        self._nfeval += 1

        # update objective values
        for i, item in enumerate(self.objectives.values()):
            item.value = f_eval[i]
        # update constraint values
        for i, item in enumerate(chain(self.e_constraints.values(),
                                       self.i_constraints.values())):
            item.value = g_eval[i]

        f = [obj.value for obj in self.objectives.values()]
        g = [con.value for con in chain(self.e_constraints.values(),
                                        self.i_constraints.values())]

        text = self._get_eval_info(f, g, False)
        opt_logger.info(text)
        print(text)

        return f, g, False

    def _get_eval_info(self, f, g, is_failed):
        """Optimization result after each step."""
        text = '{:05d} - '.format(self._nfeval)
        text += "obj(s): "
        for v in f:
            text += '{:11.4e}, '.format(v)
        if g:
            text += "con(s): "
        for v in g:
            text += '{:11.4e}, '.format(v)
        text += "var(s): "
        for value in self._x_map.values():
            text += '{:11.4e}, '.format(value)
        text += "Failed" if is_failed is True else "Succeeded"
        return text

    def solve(self, optimizer, *args, **kwargs):
        """Run the optimization and print the result.

        :param Optimizer optimizer: Optimizer instance.
        """
        text = self.summarize(False) \
               + "\n***with***\n" \
               + str(optimizer)
        logger.info(text)
        opt_logger.info(text)

        opt_f, opt_x, misc_info = optimizer(self)

        # Update objectives, variables, covariables, constraints
        # with the optimized values.
        self.eval_objs_cons(opt_x, *args, **kwargs)

        # verify solution
        f = [item.value for item in self.objectives.values()]
        np.testing.assert_almost_equal(f, opt_f)

        text = "\n" + self.summarize(True) + "\n" + misc_info
        logger.info(text)
        opt_logger.info(text)

        return opt_f, opt_x

    def summarize(self, is_solution):
        text = '\n' + '=' * 80 + '\n'
        if is_solution is False:
            text += 'Optimization problem: %s\n' % self.name
        else:
            text += 'Solution for optimization problem: %s\n' % self.name
        text += self.__str__()
        text += '\n' + '=' * 80 + '\n'
        return text

    @staticmethod
    def _format_item(instance_set, name):
        """Return structured list of parameters.

        :param OrderedDict instance_set: Should be either self.objectives,
               self.e_constraints, self.i_constraints or self.variables.
        :param str name: Name of the instance set.
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

    def __str__(self):
        text = self._format_item(self.objectives, 'Objective(s)')
        text += self._format_item(self.e_constraints, 'Equality constraint(s)')
        text += self._format_item(self.i_constraints, 'Inequality constraint(s)')
        text += self._format_item(self.variables, 'Variable(s)')
        text += self._format_item(self.covariables, 'Covariable(s)')
        return text


class LinacOptimization(Optimization):
    """Inherited from Optimization."""
    def __init__(self, linac, *, max_nf=20, **kwargs):
        """Initialization.

        :param Linac linac: Linac instance.
        :param int max_nf: Max number of allowed successive failures of of
                           calling Linac.update() method. Default = 20.
        """
        super().__init__(**kwargs)

        self._linac = linac
        # No. of consecutive failures of calling Linac.update() method
        self._nf = 0
        self._max_nf = max_nf

    def solve(self, optimizer, *args, **kwargs):
        """Run the optimization and print the result.

        Override.
        """
        logger.info(str(self._linac))
        return super().solve(optimizer, *args, **kwargs)

    def eval_objs_cons(self, x, *args, **kwargs):
        """Objective-constraint function.

        Override the method in the parent class.
        """
        self._update_x_map(x)

        # Run simulations with the new input files
        is_update_failed = True
        t0 = time.perf_counter()
        t0_cpu = time.process_time()
        try:
            self._nfeval += 1
            self._linac.run(self._x_map, *args, **kwargs)
            is_update_failed = False
            self._nf = 0
        except LisoRuntimeError as e:
            self._nf += 1
            exc_type, exc_value, exc_traceback = sys.exc_info()
            logger.debug(f"{self._nfeval:05d}: " +
                         repr(traceback.format_tb(exc_traceback)) +
                         str(e))
            logger.warning(str(e))
        except Exception as e:
            self._nf += 1
            exc_type, exc_value, exc_traceback = sys.exc_info()
            logger.error(f"{self._nfeval:05d} (Unexpected exceptions): " +
                         repr(traceback.format_tb(exc_traceback)) +
                         str(e))
            raise
        finally:
            if self._nf > self._max_nf:
                logger.info("Maximum allowed number of successive failures "
                            "reached!")

        if is_update_failed is False:
            # update objective and constraint values
            for item in chain(self.objectives.values(),
                              self.e_constraints.values(),
                              self.i_constraints.values()):
                if item.func is not None:
                    item.value = item.func(self._linac)
                else:
                    bl, src, ppt = item.expr
                    if src in ('max', 'min', 'start', 'end', 'ave', 'std'):
                        item.value = self._linac[bl].__getattribute__(
                            src).__getattribute__(ppt) * item.scale
                    else:
                        beam_params = getattr(self._linac[bl], src)
                        item.value = getattr(beam_params, ppt) * item.scale

            f = [obj.value for obj in self.objectives.values()]
            g = [con.value for con in chain(self.e_constraints.values(),
                                            self.i_constraints.values())]

            self._nf = 0
        else:
            f = [math.inf] * len(self.objectives)
            g = [math.inf] * (len(self.i_constraints) + len(self.e_constraints))

        # monitor time consumption
        dt = time.perf_counter() - t0
        dt_cpu = time.process_time() - t0_cpu
        logger.debug('{:05d}: elapsed time: {:.4f} s, cpu time: {:.4f} s'
                     .format(self._nfeval, dt, dt_cpu))

        # optimization result after each step
        text = self._get_eval_info(f, g, is_update_failed)
        logger.info(text)
        opt_logger.info(text)

        return f, g, is_update_failed
