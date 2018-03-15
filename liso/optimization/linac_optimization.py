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
    def __init__(self, linac, obj_func, *, name=None, time_out=INF,
                 max_successive_failure_allowed=20):
        """Initialization.

        :param max_successive_failure_allowed: int
            Max number of allowed successive failures.
        """
        self.linac = linac
        self.obj_func = obj_func

        if name is None:
            self.name = 'Not specified'
        else:
            self.name = name

        self.variables = OrderedDict()
        self.covariables = dict()
        self.objectives = OrderedDict()
        self.e_constraints = OrderedDict()
        self.i_constraints = OrderedDict()

        self.walkers = 1
        self.time_out = time_out

        self._nf = 0
        self._max_successive_failure_allowed = max_successive_failure_allowed

        self.start_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    def solve(self, optimizer, *, workers=1):
        """Run the optimization and print the result.

        :param optimizer: Optimizer object.
            Optimizer.
        :param workers: int
            Number of threads.
        """
        self.threads = threads
        print(self.__str__())

        t0 = datetime.now()
        optimizer(self)
        dt = datetime.now() - t0

        print(self.__str__())

    def eval_obj_func(self, x):
        """Objective-constraint function.

        This method will be called in the f_obj_con function of the optimizer.

        :param x: array-like
            Variables updated after each iteration.
        """
        # generate the variable mapping - {name: value}
        x_mapping = {key: value for key, value in zip(self.variables.keys(), x)}
        for covar in self.covariables:
            x_mapping[covar.name] = covar.value

        # Run simulations with the new input files

        is_update_failed = True

        try:
            self.linac.update(x_mapping, self.workers)
            is_update_failed = False
            self._nf = 0
        # exception propagates from Beamline.simulate() method
        except SimulationNotFinishedProperlyError as e:
            print("Simulate did not start or finish normally!")
            self._nf += 1
        # exception propagates from BeamParameters.update() method
        except FileNotFoundError as e:
            print("file not found")
            self._nf += 1
        except BeamlineMonitorError:
            raise
        except Exception as e:
            print("Unknown bug detected!")
            raise
        finally:
            if self._nf > self._max_successive_failure_allowed:
                raise SimulationSuccessiveFailureError(
                    "Maximum allowed number of successive failures reached!")

        if is_update_failed is False:
            try:
                f, g = self.obj_func(self.linac.beamlines)
                self._nf = 0
            except AttributeError:
                raise AttributeError("Tried to access a non-existed beamline "
                                     "monitor in the user-defined obj_func: "
                                     "%s!" % e)
        else:
            f = INF
            g = [INF] * (len(self.i_constraints) + len(self.e_constraints))

        return f, g, is_update_failed

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

    def add_obj(self, name):
        """Add an objective."""
        try:
            self.objectives[name] = Objective(name)
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

    @staticmethod
    def _add_instance(instance_set, instance_type, *args, **kwargs):
        """Add an instance_type to instance_set.

        :param instance_set: OrderedDict
            Should be either self.objectives, self.constraints or
            self.variables.
        :param instance_type: object
            Should be either Objective, Constraint or Variable.
        """
        if len(args) > 0:
            if isinstance(args[0], instance_type):
                instance = args[0]
                instance_set[instance.name] = instance
            else:
                name = args[0]
                try:
                    instance_set[name] = instance_type(*args, **kwargs)
                except ValueError:
                    raise

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
