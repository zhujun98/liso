#!/usr/bin/python

"""
A PYTHON script for optimizing linac.

Optimizers (SDPEN, ALPSO, NSGA2) in pyOpt are used in this
script to solve general constrained nonlinear optimization problems:

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
from datetime import datetime

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
                 name='Not specified',
                 time_out=INF,
                 max_successive_failure_allowed=20,
                 workers=1):
        """Initialization.

        :param linac: Linac object
            Linac instance.
        :param time_out: float
            Maximum time for killing the simulation.
        :param max_successive_failure_allowed: int
            Max number of allowed successive failures.
        :param workers: int
            Number of threads.
        """
        if isinstance(linac, Linac):
            self._linac = linac
        else:
            raise TypeError("{} is not a Linac instance!".format(linac))

        self.name = name
        self._workers = 1
        self.workers = workers

        self.variables = OrderedDict()
        self.covariables = OrderedDict()
        self.objectives = OrderedDict()
        self.e_constraints = OrderedDict()
        self.i_constraints = OrderedDict()

        self.time_out = time_out
        self._nf = 0
        self._max_successive_failure_allowed = max_successive_failure_allowed

        self.start_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

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

        t0 = datetime.now()
        optimizer(self)
        dt = datetime.now() - t0

        print(self.__str__())

    def eval_obj_cons(self, x):
        """Objective-constraint function.

        This method will be called in the f_obj_con function of the optimizer.

        :param x: array-like
            Variables updated after each iteration.
        """
        # generate the variable mapping - {name: value}
        x_dict = {key: value for key, value in zip(self.variables.keys(), x)}
        for covar in self.covariables:
            x_dict[covar.name] = covar.value

        # Run simulations with the new input files

        is_update_failed = True
        try:
            self._linac.update(x_dict, self.workers)
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
            if self._nf > self._max_successive_failure_allowed:
                raise SimulationSuccessiveFailureError(
                    "Maximum allowed number of successive failures reached!")

        f = INF
        g = [INF] * (len(self.i_constraints) + len(self.e_constraints))
        if is_update_failed is False:
            for obj in self.objectives.values():
                self._update_obj_con(obj)
                f = obj.value

            count = 0
            for e_con in self.e_constraints.values():
                self._update_obj_con(e_con)
                g[count] = e_con.value
                count += 1

            for i_con in self.i_constraints.values():
                self._update_obj_con(i_con)
                g[count] = i_con.value
                count += 1

            self._nf = 0

        if self._DEBUG is True:
            print('{:11.4e}'.format(f), ['{:11.4e}'.format(v) for v in g], is_update_failed)

        return f, g, is_update_failed

    def _update_obj_con(self, item):
        """Update the value of a Objective / Constraint.

        item: Objective / Constraint
            An Objective / Constraint instance.
        """
        if item.func is not None:
            item.value = item.func(self._linac)
            return

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
        except (TypeError, ValueError):
            print("Input is not valid for a Variable class instance!")
            raise

    def del_var(self, name):
        """Delete a variable by name."""
        try:
            del self.variables[name]
        except KeyError:
            raise KeyError("{} is not a variable!".format(name))

    def add_covar(self, name, *args, **kwargs):
        """Add a covariable."""
        try:
            self.covariables[name] = Covariable(name, *args, **kwargs)
        except (TypeError, ValueError):
            print("Input is not valid for a Covariable class instance!")
            raise

    def del_covar(self, name):
        """Delete a covariable by name."""
        try:
            del self.covariables[name]
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
        text = '\nOptimization Problem: %s\n' % self.name
        text += 'Started at: ' + self.start_time
        text += "\n" + '='*80
        text += self._format_output(self.objectives, 'Objectives')
        text += self._format_output(self.e_constraints, 'Equality constraints')
        text += self._format_output(self.i_constraints, 'Inequality constraints')
        text += self._format_output(self.variables, 'Variables')
        return text

    @staticmethod
    def _format_output(instance_set, title):
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
