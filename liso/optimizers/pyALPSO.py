#!/usr/bin/env python
"""
pyALPSO - An LISO interface for Augmented Larangian Particle Swarm Optimization.

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
        """Initialization"""
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
        self.rtol = 1e-3
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

        # F. Bergh, A.P. Engelbrecht,
        # A new locally convergent particle swarm optimiser,
        # Systems, Man and Cybernetics, 2002 IEEE International Conference.
        self.use_gcpso = True
        # Number of Consecutive Successes in Finding New Best Position
        # of Best Particle Before Search Radius will be Increased (GCPSO)
        self._ns = 15
        # Number of Consecutive Failures in Finding New Best Position
        # of Best Particle Before Search Radius will be Increased (GCPSO)
        self._nf = 5
        # Maximum search radius (GCPSO)
        self._rho_max = 5e-2
        # Minimum search radius (GCPSO)
        self._rho_min = 1e-4

    def __call__(self, opt_prob):
        """Run Optimizer (Optimize Routine)

        :param opt_prob:
            Optimization instance.
        """
        n_vars = len(opt_prob.variables)
        x_min = np.zeros(n_vars, float)
        x_max = np.zeros(n_vars, float)
        for i, key in enumerate(opt_prob.variables.keys()):
            x_min[i] = opt_prob.variables[key].lb
            x_max[i] = opt_prob.variables[key].ub

        def f_obj_con(x):
            f, g, _ = opt_prob.eval_objs_cons(x*(x_max - x_min) + x_min)
            return f[0], g

        # =====================================================================

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
        opt_x, opt_L, opt_f, opt_g, opt_lambda, opt_rp, k_out, nfeval, stop_info = \
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
                  self.use_gcpso,
                  self._nf,
                  self._ns,
                  self._rho_max,
                  self._rho_min,
                  f_obj_con
                  )
        opt_x[:] = opt_x*(x_max - x_min) + x_min
        delta_t = time.time() - t0

        if self.printout > 0:
            self._print_title(opt_prob.name)

            print(self.__str__())
            print("No. of outer iterations: %d" % k_out)
            print("No. of objective function evaluations: %d" % nfeval)

            text = ''

            text += "\nBest position:\n"
            for j in range(n_vars):
                text += ("    P(%d) = %11.4e" % (j, opt_x[j]))
                if np.mod(j + 1, 3) == 0 and j != n_vars - 1:
                    text += "\n"
            text += "\n"

            text += "\nObjective function value:\n"
            text += "    F = %11.4e\n" % opt_f

            if n_cons > 0:
                text += "\nAugmented Lagrangian function value:\n"
                text += "    L = %11.4e\n" % opt_L

                if n_eq_cons > 0:
                    text += "\nEquality constraint violation value(s):\n"
                    for j in range(n_eq_cons):
                        text += "    H(%d) = %11.4e" % (j, opt_g[j])
                    text += "\n"

                if n_cons > n_eq_cons:
                    text += "\nInequality constraint violation value(s):\n"
                    for j in range(n_eq_cons, n_cons):
                        text += "    G(%d) = %11.4e" % (j, opt_g[j])
                    text += "\n"

                text += "\nLagrangian multiplier value(s):\n"
                for j in range(n_cons):
                    text += "    M(%d) = %11.4e" % (j, opt_lambda[j])
                    if np.mod(j + 1, 3) == 0 and j != n_cons - 1:
                        text += "\n"
                text += "\n"

                text += "\nPenalty factor value(s):\n"
                for j in range(n_cons):
                    text += "    R(%d) = %11.4e" % (j, opt_rp[j])
                    if np.mod(j + 1, 3) == 0 and j != n_cons - 1:
                        text += "\n"
                text += "\n"

            print(text)

            self._print_additional_info([stop_info])

        return opt_f, opt_x, {'time': delta_t}

    def __str__(self):
        text = '-' * 80 + '\n'
        text += 'Augmented Lagrangian Particle Swarm Optimizer (ALPSO)\n'
        text += '-' * 80 + '\n'

        text += 'Swarm size       : %7d' % self.swarm_size + \
                '  Equality tol.   : %7.1e' % self.etol + \
                '  Max. outer iter.: %7d\n' % self.max_outer_iter
        text += 'Cognitive param. : %7.2f' % self.c1 + \
                '  Inequality tol. : %7.1e' % self.itol + \
                '  Max. inner iter.: %7d\n' % self.max_inner_iter
        text += 'Social param.    : %7.2f' % self.c2 + \
                '  Relative tol.   : %7.1e' % self.rtol + \
                '  Min. inner iter.: %7d\n' % self.min_inner_iter
        text += 'Initial weight   : %7.2f' % self.w0 + \
                '  Absolute tol.   : %7.1e' % self.atol + \
                '  Topology        : %7s\n' % self.topology
        text += 'Final weight     : %7.2f' % self.w1 + \
                '  Divergence tol. : %7.1e\n' % self.dtol

        text += '-' * 80 + '\n\n'

        return text
