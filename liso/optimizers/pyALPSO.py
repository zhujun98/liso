"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import time

import numpy as np

from .optimizer import Optimizer
from .alpso import alpso


class ALPSO(Optimizer):
    """ALPSO Optimizer Class.

    Attributes:
        swarm_size (int): Number of particles. Default = 40.
        topology (str): Topology of the swarm. Default = 'gbest'.
        max_outer_iter (int): Maximum Number of Outer Loop Iterations.
                              Default = 6.
        max_inner_iter (int): Maximum Number of Inner Loop Iterations.
                              Default = 3.
        min_inner_iter (int): Minimum Number of Inner Loop Iterations.
        etol (float): Absolute tolerance for equality constraints.
                      Default = 1e-4.
        itol (float): Absolute tolerance for inequality constraints.
                      Default = 1e-4.
        rtol (float): Relative tolerance for Lagrange function. Default = 1e-3.
        atol (float): Absolute tolerance for Lagrange function. Default = 1e-4.
        dtol (float): Absolute tolerance for position deviation of all particles
                      . Default = 5e-2.
        c1 (float): Cognitive Parameter. Default = 1.5.
        c2 (float): Social Parameter. Default = 1.5.
        w0 (float): Initial Inertia Weight. Default = 0.90.
        w1 (float): Final Inertia Weight. Default = 0.40.
        use_gcpso (bool): Use Guaranteed Convergence Particle Swarm Optimization
                          (F. Bergh, A.P. Engelbrecht, A new locally convergent
                          particle swarm optimiser, Systems, Man and Cybernetics
                          , 2002 IEEE International Conference). Default = True.

    """
    category = 'global'
    name = 'ALPSO'
    multiprocessing = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # -------------------------------------------------------------
        # default optimizer settings
        # -------------------------------------------------------------

        self.swarm_size = 40
        self.topology = 'gbest'
        self.max_outer_iter = 100
        self.max_inner_iter = 6
        self.min_inner_iter = 3

        self.etol = 1e-3
        self.itol = 1e-3
        self.rtol = 1e-2
        self.atol = 1e-4
        self.dtol = 1e-1

        self.c1 = 1.5
        self.c2 = 1.5
        self.w0 = 0.90
        self.w1 = 0.40

        #
        self.use_gcpso = True
        # Number of Consecutive Successes in Finding New Best Position
        # of Best Particle Before Search Radius will be Increased (GCPSO)
        self._ns = 15
        # Number of Consecutive Failures in Finding New Best Position
        # of Best Particle Before Search Radius will be Increased (GCPSO)
        self._nf = 5
        # Maximum search radius (GCPSO)
        self._rho_max = 0.5
        # Minimum search radius (GCPSO)
        self._rho_min = 0.0001

    def __call__(self, opt_prob):
        """Run Optimizer (Optimize Routine)

        Override.
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
        np.random.seed(self._seed)  # Set random number generator
        x0 = np.random.rand(self.swarm_size, n_vars)
        v0 = np.zeros([self.swarm_size, n_vars], float)

        t0 = time.perf_counter()
        opt_x, opt_f, k_out, nfeval, stop_info = \
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
        delta_t = time.perf_counter() - t0

        # miscellaneous information
        misc_info = ""
        misc_info += "%s\n\n" % stop_info
        misc_info += "No. of objective function evaluations: %d\n" % nfeval
        misc_info += "Time consumed: %f second(s)\n" % delta_t
        misc_info += "No. of outer iteration(s): %d\n" % k_out

        return opt_f, opt_x, misc_info

    def __str__(self):
        text = '\n' + '-' * 80 + '\n'
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
