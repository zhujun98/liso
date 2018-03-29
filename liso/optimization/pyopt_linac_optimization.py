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
from .linac_optimization import LinacOptimization
from pyOpt import Optimization
from ..config import Config
from ..simulation.simulation_utils import check_templates

INF = Config.INF


class PyoptLinacOptimization(LinacOptimization):
    """LinacOpt class."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def solve(self, optimizer):
        """Run the optimization and print the result.

        Override the method in the parent class.

        :param optimizer: Optimizer object.
            Optimizer.
        """
        check_templates(self._linac.get_templates(), self._x_map)

        print(self.__str__())
        # TODO::check whether the optimizer and opt_prob match?
        opt_prob = self._adapt_optimization()
        opt_f, opt_x, _ = optimizer(opt_prob)
        self._create_solution(opt_x)
        self._verify_solution(opt_f)
        print(self.__str__())

    def _adapt_optimization(self):
        """Set an Optimization instance used by pyOpt."""
        opt_prob = Optimization("opt_prob", self.eval_obj_cons)

        # pyOpt relies on the str(int) type key for variables, constraints,
        # so that inside the dictionary the items are sorted by key.
        for var in self.variables.values():
            opt_prob.addVar(var.name, var.type_,
                            lower=var.lb,
                            upper=var.ub,
                            value=var.value)
        for obj in self.objectives.values():
            opt_prob.addObj(obj.name)
        # all equality constraints must be added before inequality constraints.
        for ec in self.e_constraints.values():
            opt_prob.addCon(ec.name, 'e')
        for ic in self.i_constraints.values():
            opt_prob.addCon(ic.name, 'i')

        return opt_prob
