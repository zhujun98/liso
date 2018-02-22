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
from datetime import datetime

from liso.optimization.linac_optimization import LinacOptimization
from pyOpt import Optimization
from liso.backend import config

INF = config['INF']


class PyoptLinacOptimization(LinacOptimization):
    """LinacOpt class."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def solve(self, optimizer, *, threads=1):
        """Run the optimization and print the result.

        Override the method in the parent class.

        :param optimizer: Optimizer object.
            Optimizer.
        :param threads: int
            Number of threads.
        """
        self.threads = threads
        print(self.__str__())

        opt_prob = Optimization("opt_prob", self.eval_obj_func)
        # Convert variables, constraints and object in API to pyOpt
        #
        # pyOpt relies on the str(int) type key for variables, constraints,
        # so that inside the dictionary the items are sorted by key.
        for var in self.variables.values():
            opt_prob.addVar(var.name, var.type_,
                            lower=var.lower, upper=var.upper, value=var.value)

        for obj in self.objectives.values():
            opt_prob.addObj(obj.name)

        for ec in self.e_constraints.values():
            opt_prob.addCon(ec.name, 'e')
        for ic in self.i_constraints.values():
            opt_prob.addCon(ic.name, 'i')

        # TODO::check whether the optimizer and opt_prob match?
        # Run optimization
        t0 = datetime.now()
        optimizer(opt_prob)
        dt = datetime.now() - t0

        # Paste the solution in pyOpt to this API
        for var in opt_prob.solution(0).getVarSet().values():
            self.variables[var.name].value = var.value

        for obj in opt_prob.solution(0).getObjSet().values():
            self.objectives[obj.name].value = obj.value

        count = 0
        for con in opt_prob.solution(0).getConSet().values():
            count += 1
            if count <= len(self.e_constraints):
                self.e_constraints[con.name].value = con.value
            else:
                self.i_constraints[con.name].value = con.value

        print(self.__str__())
