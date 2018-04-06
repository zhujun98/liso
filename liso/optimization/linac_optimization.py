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

from ..operation import Operation
from .variable import Variable
from ..covariable import Covariable
from .constraint import EConstraint
from .constraint import IConstraint
from .objective import Objective
from ..config import Config
from ..exceptions import *
from ..simulation.simulation_utils import check_templates

INF = Config.INF


class Optimization(Operation):
    """Inherited from Operation."""
    def __init__(self, name='unnamed', *, opt_func=None):
        """Initialization.

        :param name: str
            Name of the optimization problem (arbitrary).
        :param opt_func: callable
            A callable object which returns (objective, constraints).
        """
        super().__init__(name)

        self.variables = OrderedDict()
        self.covariables = OrderedDict()
        self.objectives = OrderedDict()
        self.e_constraints = OrderedDict()
        self.i_constraints = OrderedDict()

        self._x_map = OrderedDict()  # Initialize the x_map dictionary

        self._nfeval = 0  # No. of evaluations on the objectives and constraints.

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

    def _update_objs_cons(self):
        """Update all Objective and Constraints.

        Invoked in self.eval_objs_cons().
        """
        x = list(self._x_map.values())
        f_eval, g_eval = self._opt_func(x)

        for i, item in enumerate(self.objectives.values()):
            item.value = f_eval[i]

        for i, item in enumerate(chain(self.e_constraints.values(), self.i_constraints.values())):
            item.value = g_eval[i]

        f, g = self._get_objs_cons()

        return f, g

    def eval_objs_cons(self, x):
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
        f, g = self._update_objs_cons()

        return f, g, False

    def solve(self, optimizer):
        """Run the optimization and print the result.

        :param optimizer: Optimizer object.
            Optimizer.
        """
        if self.printout > 0:
            print(optimizer)

        print(self.__str__())

        opt_f, opt_x, _ = optimizer(self)

        # Update objectives, variables, covariables, constraints
        # with the optimized values.
        self.eval_objs_cons(opt_x)

        self._verify_solution(opt_f)

        print(self.__str__())

    def _get_objs_cons(self):
        """Get the values of all objectives and constraints."""
        f = []
        for obj in self.objectives.values():
            f.append(obj.value)

        g = []
        for con in chain(self.e_constraints.values(), self.i_constraints.values()):
            g.append(con.value)

        return f, g

    def _verify_solution(self, opt_f):
        """Verify the solution.

        :param opt_f: list
            Optimized objective value(s).
        """
        f = [item.value for item in self.objectives.values()]
        np.testing.assert_almost_equal(f, opt_f)

    @staticmethod
    def _format_item(instance_set, name):
        """Return structured list of parameters.

        :param instance_set: OrderedDict
            Should be either self.objectives, self.constraints or
            self.variables.
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

    def __str__(self):
        text = '\nOptimization Problem: %s' % self.name
        text += '\n' + '='*80 + '\n'
        text += self._format_item(self.objectives, 'Objective(s)')
        text += self._format_item(self.e_constraints, 'Equality constraint(s)')
        text += self._format_item(self.i_constraints, 'Inequality constraint(s)')
        text += self._format_item(self.variables, 'Variable(s)')
        text += self._format_item(self.covariables, 'Covariable(s)')
        return text


class LinacOptimization(Optimization):
    """Inherited from Optimization."""
    def __init__(self, linac, name='unnamed', *, max_nf=20):
        """Initialization.

        :param linac: Linac instance
            Linac instance.
        :param max_nf: int
            Max number of allowed successive failures of of calling
            Linac.update() method.
        """
        super().__init__(name)

        self._linac = linac
        # No. of consecutive failures of calling Linac.update() method
        self._nf = 0
        self._max_nf = max_nf

    def solve(self, optimizer):
        """Run the optimization and print the result.

        Override the method in the parent class.
        """
        check_templates(self._linac.get_templates(), self._x_map)
        super().solve(optimizer)

    def eval_objs_cons(self, x):
        """Objective-constraint function.

        Override the method in the parent class.
        """
        self._update_x_map(x)

        # Run simulations with the new input files
        is_update_failed = True
        dt = 0.0
        dt_cpu = 0.0
        try:
            self._nfeval += 1
            dt, dt_cpu = self._linac.update(self._x_map, self.workers)
            is_update_failed = False
            self._nf = 0
        # exception propagates from Beamline.simulate() method
        except SimulationNotFinishedProperlyError as e:
            print(e)
        # exception propagates from Beamline.update() method
        except WatchUpdateFailError as e:
            self._nf += 1
            print("Watch update failed: {}".format(e))
        # exception propagates from Beamline.update() method
        # Note: In practice, only WatchUpdateFailError could be raised since
        # Watch is updated before Line!
        except LineUpdateFailError as e:
            self._nf += 1
            print("Line update failed: {}".format(e))
        except Exception as e:
            self._nf += 1
            print("Unknown exceptions:\n {}".format(e))
            raise
        finally:
            if self._nf > self._max_nf:
                raise SimulationSuccessiveFailureError(
                    "Maximum allowed number of successive failures reached!")

        if is_update_failed is False:
            f, g = self._update_objs_cons()
            self._nf = 0
        else:
            f = [INF] * len(self.objectives)
            g = [INF] * (len(self.i_constraints) + len(self.e_constraints))

        if self.monitor_time is True:
            print('elapsed time: {:.4f} s, '.format(dt),
                  'cpu time: {:.4f} s, '.format(dt_cpu))

        if self.printout > 0:
            print('{:04d} - '.format(self._nfeval),
                  ['{:^12}: {:9.2e}'.format(key, value) for key, value in self._x_map.items()],
                  ['{:9.2e}'.format(v) for v in f],
                  ['{:9.2e}'.format(v) for v in g],
                  is_update_failed)

        return f, g, is_update_failed

    def _update_objs_cons(self):
        """Update all Objectives and Constraints.

        Override the method in the parent class.
        """
        for item in chain(self.objectives.values(),
                          self.e_constraints.values(),
                          self.i_constraints.values()):
            if item.func is not None:
                item.value = item.func(self._linac)
            else:
                keys = item.expr
                if keys[1] in ('max', 'min', 'start', 'end', 'ave', 'std'):
                    item.value = self._linac.__getattr__(keys[0]).__getattribute__(
                        keys[1]).__getattribute__(keys[2]) * item.scale
                else:
                    item.value = self._linac.__getattr__(keys[0]).__getattr__(
                        keys[1]).__getattribute__(keys[2]) * item.scale

        f, g = self._get_objs_cons()

        return f, g
