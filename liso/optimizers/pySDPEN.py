"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import time

from pyOpt import SDPEN as pyoptSDPEN

from .pyopt_optimizer import PyoptOptimizer
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


class SDPEN(PyoptOptimizer):
    """SDPEN optimizer class."""
    category = 'local'
    name = 'pyOpt_SDPEN'
    multiprocessing = False

    rtol = SDPENOption('alfa_stop')
    max_iter = SDPENOption('nf_max')

    # False for turning off the support inequality constraint
    _constraint_on = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._sdpen = pyoptSDPEN()
        self._sdpen.setOption('iprint', -1)

        self.rtol = 1.0e-6
        self.max_iter = 5000

    def __call__(self, opt_prob, workers=1):
        """Run Optimizer (Optimize Routine)

        Override.
        """
        if self.__class__.__dict__['_constraint_on'] is True:
            # Constraints
            if opt_prob.e_constraints:
                raise OptimizationConstraintSupportError(
                    "{} optimizer does not support equality constraints!"
                    .format(self.name))
        else:
            # The implementation of SDPEN in pyOpt actually supports
            # inequality constraint, but the performance is very poor
            # (see tests/test_sdpen). Therefore, we suspend the
            # constraint in LISO.
            if opt_prob.e_constraints or opt_prob.i_constraints:
                raise OptimizationConstraintSupportError(
                    "{} optimizer does not support constraints!"
                    .format(self.name))

        pyopt_prob = self._to_pyopt_optimization(opt_prob)

        t0 = time.perf_counter()
        opt_f, opt_x, pyopt_info = self._sdpen(pyopt_prob)
        opt_f = opt_f[0]  # pyOpt.Optimizer.__call__() returns a list
        delta_t = time.perf_counter() - t0

        # miscellaneous information
        misc_info = ""
        misc_info += "%s\n\n" % pyopt_info['text']
        misc_info += "Time consumed: %f second(s)\n" % delta_t

        return opt_f, opt_x, misc_info

    def __str__(self):
        text = '\n' + '-' * 80 + '\n'
        text += 'Sequential Penalty Derivative-free Method (SDPEN)\n'
        text += '-' * 80 + '\n'

        text += 'Max. iter.     : %7d' % self.max_iter + \
                '  Relative tol.: % 7.1e\n' % self.rtol

        text += '-' * 80 + '\n'

        return text
