"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import time

import numpy as np

from .optimizer import Optimizer
from .nelder_mead import nelder_mead
from ..exceptions import OptimizationConstraintSupportError


class NelderMead(Optimizer):
    """NelderMead Optimizer Class.

    Attributes:
        rtol (float): Relative tolerance for Lagrange function. Default = 1e-3.
        atol (float): Absolute tolerance for Lagrange function. Default = 1e-4.
        max_stag (int): Maximum number of stagnation (no improvement).
                        Default = 10.
        max_iter (int): Maximum number of iterations. Default = 10000.

    """
    category = 'local'
    name = 'Nelder-Mead'
    multiprocessing = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # -------------------------------------------------------------
        # default optimizer settings
        # -------------------------------------------------------------

        self.rtol = 1e-3
        self.atol = 1e-4
        self.max_stag = 10
        self.max_iter = 10000

        self._alpha = 1.0  # reflection coefficient (0.0, INF)
        self._gamma = 2.0  # expansion coefficient: (1.0, INF)
        self._beta = 0.5  # contraction coefficient: (0, 0.5]
        self._sigma = 0.5  # shrink coefficient

        # Relative change of position at initialization when it is not zero
        self._relative_delta = 0.05
        # Absolution change of position at initialization when it is zero
        self._absolute_delta = 0.00025

    def __call__(self, opt_prob, workers=1):
        """Run Optimizer (Optimize Routine)

        Override.
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

        return opt_f, opt_x, misc_info

    def __str__(self):
        text = '\n' + '-' * 80 + '\n'
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
