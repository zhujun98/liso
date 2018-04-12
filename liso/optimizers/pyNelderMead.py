#!/usr/bin/env python
"""
pyNelderMead - An LISO interface for Nelder-Mead (simplex) algorithm.

Author: Jun Zhu
"""
import time

import numpy as np

from .optimizer import Optimizer
from .nelder_mead import nelder_mead
from ..exceptions import OptimizationConstraintSupportError


class NelderMead(Optimizer):
    """NelderMead Optimizer Class.

    Inherited from Optimizer Abstract Class."""
    category = 'local'

    def __init__(self):
        """Initialization"""
        name = 'Nelder-Mead'
        super().__init__(name)

        # -------------------------------------------------------------
        # default optimizer settings
        # -------------------------------------------------------------

        # Relative tolerance for Lagrange function
        self.rtol = 1e-3
        # Absolute tolerance for Lagrange function
        self.atol = 1e-4
        # Maximum number of stagnation (no improvement)
        self.max_stag = 10
        # Maximum number of iterations
        self.max_iter = 10000

        # reflection coefficient (0.0, INF)
        self._alpha = 1.0
        # expansion coefficient: (1.0, INF)
        self._gamma = 2.0
        # contraction coefficient: (0, 0.5]
        self._beta = 0.5
        # shrink coefficient
        self._sigma = 0.5

        # Relative change of position at initialization when it is not zero
        self._relative_delta = 0.05
        # Absolution change of position at initialization when it is zero
        self._absolute_delta = 0.00025

    def __call__(self, opt_prob):
        """Run Optimizer (Optimize Routine)

        :param opt_prob:
            Optimization instance.
        """
        # Constraints
        if opt_prob.e_constraints or opt_prob.i_constraints:
            raise OptimizationConstraintSupportError(
                "{} optimizer does not support constraint!".format(self.name))

        n_vars = len(opt_prob.variables)
        x_min = np.zeros(n_vars, float)
        x_max = np.zeros(n_vars, float)
        x_init = np.zeros(n_vars, float)
        for i, key in enumerate(opt_prob.variables.keys()):
            x_init[i] = opt_prob.variables[key].value
            x_min[i] = opt_prob.variables[key].lb
            x_max[i] = opt_prob.variables[key].ub

        def f_obj(x):
            f, _, _ = opt_prob.eval_objs_cons(x*(x_max - x_min) + x_min)
            return f[0]

        # =====================================================================

        # =============================================================
        # Run Nelder-Mead
        # =============================================================

        # Initialize the simplex vertices
        x0 = np.zeros((n_vars + 1, n_vars))
        for i in range(n_vars+1):
            x0[i, :] = x_init[:]
            if i == 0:
                continue
            if abs(x0[i, i-1]) < self._absolute_delta / self._relative_delta:
                x0[i, i-1] += self._absolute_delta
            else:
                x0[i, i-1] += self._relative_delta * x0[i, i-1]
            # check fitness
            x0[i, i - 1] = min(x_max[i - 1], max(x_min[i - 1], x0[i, i - 1]))

        x0_normalized = (x0 - x_min) / (x_max - x_min)  # normalize to [0, 1]

        t0 = time.perf_counter()
        opt_x, opt_f, k_iter, nfeval, k_misc, stop_info = \
            nelder_mead(x0_normalized,
                        self.rtol,
                        self.atol,
                        self.max_stag,
                        self.max_iter,
                        self._alpha,
                        self._gamma,
                        self._beta,
                        self._sigma,
                        f_obj)
        opt_x[:] = opt_x*(x_max - x_min) + x_min
        delta_t = time.perf_counter() - t0

        # miscellaneous information
        misc_info = ""
        misc_info += "%s\n\n" % stop_info
        misc_info += "No. of objective function evaluations: %d\n" % nfeval
        misc_info += "Time consumed: %f second(s)\n" % delta_t
        misc_info += "No. of iteration(s): %d\n" % k_iter
        misc_info += "No. of reflection operation(s): %d\n" % k_misc[0]
        misc_info += "No. of expansion operation(s): %d\n" % k_misc[1]
        misc_info += "No. of inner contraction operation(s): %d\n" % k_misc[2]
        misc_info += "No. of outer contraction operation(s): %d\n" % k_misc[3]
        misc_info += "No. of shrink operation(s): %d\n" % k_misc[4]

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
        text += 'Nelder-Mead (Simplex) optimizer\n'
        text += '-' * 80 + '\n'

        text += 'Rel. delta   : %7.2e' % self._relative_delta + \
                '  Max. stag.    : %7d' % self.max_stag + \
                '  Reflection coeff.  : %7.4f\n' % self._alpha
        text += 'Abs. delta   : %7.2e' % self._absolute_delta + \
                '  Max. iter.    : %7d' % self.max_iter + \
                '  Expansion coeff.   : %7.4f\n' % self._gamma
        text += '                       ' + \
                '  Relative tol. : %7.1e' % self.rtol + \
                '  Contraction coeff. : %7.4f\n' % self._beta
        text += '                       ' + \
                '  Absolute tol. : %7.1e' % self.atol + \
                '  Shrink coeff.      : %7.4f\n' % self._sigma

        text += '-' * 80 + '\n\n'

        return text
