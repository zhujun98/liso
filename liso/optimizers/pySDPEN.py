#!/usr/bin/python
"""
pyNelderMead - An LISO interface for optimizer - pyOpt.SDPEN.

Author: Jun Zhu
"""
import time
import numpy as np

from pyOpt import SDPEN as pyoptSDPEN

from .optimizer import Optimizer
from ..optimization.pyopt_adapter import to_pyopt_optimization
from ..exceptions import OptimizationConstraintSupportError


class SDPENOption(object):
    """An interface for setting the SDPEN options."""
    def __init__(self, name):
        """Initialization.

        :param name: str
            Name of the option.
        """
        self.name = name

    def __get__(self, instance, owner):
        return instance.__dict__['_sdpen'].getOption(self.name)

    def __set__(self, instance, value):
        return instance.__dict__['_sdpen'].setOption(self.name, value)


class SDPEN(Optimizer):
    """SDPEN optimizer class."""
    category = 'local'

    rtol = SDPENOption('alfa_stop')
    max_iter = SDPENOption('nf_max')

    # False for turning off the support inequality constraint
    _constraint_on = False

    def __init__(self):
        """Initialization."""
        name = 'pyOpt_SDPEN'
        super().__init__(name)

        self._sdpen = pyoptSDPEN()
        self._sdpen.setOption('iprint', -1)

        self.rtol = 1.0e-6
        self.max_iter = 5000

    def __call__(self, opt_prob):
        """Run Optimizer (Optimize Routine)

        :param opt_prob:
            Optimization instance.
        """
        if self.__class__.__dict__['_constraint_on'] is True:
            # Constraints
            if opt_prob.e_constraints:
                raise OptimizationConstraintSupportError(
                    "{} optimizer does not support equality constraints!".format(self.name))
        else:
            # The implementation of SDPEN in pyOpt actually supports inequality constraint,
            # but the performance is very poor (see tests/test_sdpen). Therefore, we suspend
            # the constraint in LISO.
            if opt_prob.e_constraints or opt_prob.i_constraints:
                raise OptimizationConstraintSupportError(
                    "{} optimizer does not support constraints!".format(self.name))

        n_vars = len(opt_prob.variables)

        pyopt_prob = to_pyopt_optimization(opt_prob)

        t0 = time.perf_counter()
        opt_f, opt_x, pyopt_info = self._sdpen(pyopt_prob)
        opt_f = opt_f[0]  # pyOpt.Optimizer.__call__() returns a list
        delta_t = time.perf_counter() - t0

        # miscellaneous information
        misc_info = ""
        misc_info += "%s\n\n" % pyopt_info['text']
        misc_info += "Time consumed: %f second(s)\n" % delta_t

        if self.printout > 0:
            self._print_title(opt_prob.name)
            print(self.__str__())
            print(misc_info)

            text = ''

            text += "\nBest position:\n"
            for j in range(n_vars):
                text += ("    P(%d) = %11.4e" % (j, opt_x[j]))
                if np.mod(j + 1, 3) == 0 and j != n_vars - 1:
                    text += "\n"
            text += "\n"

            text += "\nObjective function value:\n"
            text += "    F = %11.4e\n" % opt_f

            print(text)

        return opt_f, opt_x, misc_info

    def __str__(self):
        text = '-' * 80 + '\n'
        text += 'Sequential Penalty Derivative-free Method (SDPEN)\n'
        text += '-' * 80 + '\n'

        text += 'Max. iter.     : %7d' % self.max_iter + \
                '  Relative tol.: % 7.1e\n' % self.rtol

        text += '-' * 80 + '\n'

        return text
