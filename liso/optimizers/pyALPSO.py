#!/usr/bin/env python
"""
pyALPSO - A Python pyOpt interface for ALPSO.

Author: Jun Zhu
"""
import time

import numpy as np

from .optimizer import Optimizer
from .alpso import alpso


class ALPSO(Optimizer):
    """ALPSO Optimizer Class.

    Inherited from Optimizer Abstract Class."""
    category = 'global'

    def __init__(self):
        """ALPSO Optimizer Class Initialization"""
        name = 'ALPSO'
        super().__init__(name)

        # -------------------------------------------------------------
        # default optimizer settings
        # -------------------------------------------------------------

        # Number of Particles (Depends on Problem dimensions)
        self.swarm_size = 40
        #
        self.topology = 'gbest'
        # Maximum Number of Outer Loop Iterations
        self.max_outer_iter = 100
        # Maximum Number of Inner Loop Iterations
        self.max_inner_iter = 6
        # Minimum Number of Inner Loop Iterations
        self.min_inner_iter = 3

        # Absolute tolerance for equality constraints
        self.etol = 1e-3
        # Absolute tolerance for inequality constraints
        self.itol = 1e-3

        # Relative tolerance for Lagrange function
        self.rtol = 2e-3
        # Absolute tolerance for Lagrange function
        self.atol = 1e-3
        # Absolute tolerance for position deviation of all particles
        self.dtol = 1e-2

        # Cognitive Parameter
        self.c1 = 1.5
        # Social Parameter
        self.c2 = 1.5
        # Initial Inertia Weight
        self.w0 = 0.90
        # Final Inertia Weight
        self.w1 = 0.40

        # Random Number Seed (None - Auto-Seed based on time clock)
        self.seed = None
        if self.seed is None:
            self.seed = int(time.time())

        self.verbose = True

    def __call__(self, opt_prob):
        """Run Optimizer (Optimize Routine)

        :param opt_prob:
            Optimization instance.
        """
        n_vars = len(opt_prob.variables)
        x_min = np.zeros(n_vars, float)
        x_max = np.zeros(n_vars, float)
        i = 0
        for key in opt_prob.variables.keys():
            x_min[i] = opt_prob.variables[key].lb
            x_max[i] = opt_prob.variables[key].ub
            i += 1

        def f_obj_con(x):
            """ALPSO - Objective/Constraint Values Function.

            :param x: np.array, [swarm_size, n_vars]
                Normalized variables (to space [0, 1]).

            :return: f: float
                Objective value.
            :return: g: numpy.array
                Constraint values.
            """
            f, g, _ = opt_prob.eval_obj_cons(x*(x_max - x_min) + x_min)
            # single objective problem
            return f[0], g

        # =====================================================================
        name = opt_prob.name

        # Constraints
        n_eq_cons = len(opt_prob.e_constraints)
        n_cons = n_eq_cons + len(opt_prob.i_constraints)

        # =============================================================
        # Run ALPSO
        # =============================================================
        np.random.seed(self.seed)  # Set random number generator
        x0 = np.random.rand(self.swarm_size, n_vars)
        v0 = np.zeros([self.swarm_size, n_vars], float)

        t0 = time.time()
        opt_x, opt_L, opt_f, opt_g, opt_lambda, opt_rp, k_out, nfevals, stop_info = \
            alpso(x0,
                  v0,
                  n_cons,
                  n_eq_cons,
                  self.topology,
                  self.max_outer_iter,
                  self.max_inner_iter,
                  self.min_inner_iter,
                  self.etol,
                  self.itol,
                  self.rtol,
                  self.atol,
                  self.dtol,
                  self.c1,
                  self.c2,
                  self.w0,
                  self.w1,
                  f_obj_con
                  )
        opt_x[:] = opt_x*(x_max - x_min) + x_min
        delta_t = time.time() - t0

        if self.verbose is True:
            # Print Results
            print("=" * 80)
            print(name + "\n")
            print(stop_info + "\n")
            print("No. of outer iterations: %d" % k_out)
            print("No. of objective function evaluations: %d" % nfevals)

            text = ''

            text += "\nBest position:\n"
            for j in range(n_vars):
                text += ("\tP(%d) = %.4e\t" % (j, opt_x[j]))
                if np.mod(j + 1, 3) == 0 and j != n_vars - 1:
                    text += "\n"

            text += "\nObjective function value:\n"
            text += "\tF = %.4e" % opt_f

            if n_cons > 0:
                text += "\nAugmented Lagrangian function value:\n"
                text += "\tL = %.4e" % opt_L

                text += "\nEquality constraint violation values:\n"
                for j in range(n_eq_cons):
                    text += "\tH(%d) = %.4e\t" % (j, opt_g[j])

                text += "\nInequality constraint violation values:\n"
                for j in range(n_eq_cons, n_cons):
                    text += "\tG(%d) = %.4e\t" % (j, opt_g[j])

                text += "\nLagrangian multiplier values:\n"
                for j in range(n_cons):
                    text += "\tL(%d) = %.4e\t" % (j, opt_lambda[j])
                    if np.mod(j + 1, 3) == 0 and j != n_vars - 1:
                        text += "\n"

                text += "\nPenalty factor values:\n"
                for j in range(n_cons):
                    text += "\tL(%d) = %.4e\t" % (j, opt_rp[j])
                    if np.mod(j + 1, 3) == 0 and j != n_vars - 1:
                        text += "\n"

            print(text + "\n" + "=" * 80 + "\n")

        return opt_f, opt_x, {'time': delta_t}

    def __str__(self):
        header = ''
        header += ' ' * 37 + '======================\n'
        header += ' ' * 39 + ' ALPSO (Serial)\n'
        header += ' ' * 37 + '======================\n\n'
        header += 'Parameters:\n'
        header += '-' * 97 + '\n'

        header += 'SwarmSize           :%8d' % self.swarm_size + \
                  '    MaxOuterIters       :%8d' % self.max_outer_iter + \
                  '    Topology            : %7s\n' % self.topology
        header += 'Cognitive Parameter :%8.2f' % self.c1 + \
                  '    MaxInnerIters       :%8d\n' % self.max_inner_iter
        header += 'Social Parameter    :%8.2f' % self.c2 + \
                  '    MinInnerIters       :%8d' % self.min_inner_iter + \
                  '    Relative Tolerance  : %7.1e\n' % self.rtol
        header += 'Initial Weight      :%8.2f' % self.w0 + \
                  '    Equality Tolerance  : %7.1e' % self.etol + \
                  '    Absolute Tolerance  : %7.1e\n' % self.atol
        header += 'Final Weight        :%8.2f' % self.w1 + \
                  '    Inequality Tolerance: %7.1e' % self.itol + \
                  '    Divergence Tolerance: %7.1e\n' % self.dtol

        header += '-' * 97 + '\n\n'

        return header
