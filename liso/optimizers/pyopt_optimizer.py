"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
from .optimizer import Optimizer
from pyOpt import Optimization as pyoptOptimization


class PyoptOptimizer(Optimizer):
    """Abstract class for optimizers in pyOpt.

    Inherited from Optimizer.
    """
    @staticmethod
    def _to_pyopt_optimization(opt_prob):
        """Adapt liso.Optimization to pyOpt.Optimization.

        :param opt_prob: liso.Optimization
            An liso.Optimization instance.
        """
        pyopt_prob = pyoptOptimization("opt_prob", opt_prob.eval_objs_cons)

        # pyOpt relies on the str(int) type key for variables, constraints,
        # so that inside the dictionary the items are sorted by key.
        for var in opt_prob.variables.values():
            pyopt_prob.addVar(var.name, var.type_,
                              lower=var.lb,
                              upper=var.ub,
                              value=var.value)
        for obj in opt_prob.objectives.values():
            pyopt_prob.addObj(obj.name)
        # all equality constraints must be added before inequality constraints.
        for ec in opt_prob.e_constraints.values():
            pyopt_prob.addCon(ec.name, 'e')
        for ic in opt_prob.i_constraints.values():
            pyopt_prob.addCon(ic.name, 'i')

        return pyopt_prob
