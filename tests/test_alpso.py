"""
Unittest of ALPSO optimizer.

Eggholder and TP37 have higher probabilities of failure.

Author: Jun Zhu, zhujun981661@gmail.com
"""
import unittest

import numpy as np

from liso import Optimization, ALPSO

from opt_problems import (
    Rastrigin, Rosenbrock, EggHolder, TP08, TP14, TP32, TP37, TP43
)


class TestALPSO(unittest.TestCase):
    def setUp(self):
        self.optimizer = ALPSO()

    def _setup_test(self, cls, *,
                    swarm_size=100, atol=None, rtol=1e-2, dtol=0.01):
        """Set up a test.

        :param cls: OptimizationTestProblem instance
            A test problem.
        :param swarm_size: int
            Swarm size for ALPSO.
        :param atol: float
            Absolute tolerance of the objective.
        :param rtol: float
            Relative tolerance of the objective.
        :param dtol: float
            Absolute tolerance of the position (L2 norm).
        """
        self.optimizer.seed = 2  # Get consistent result
        self.optimizer.swarm_size = swarm_size

        opt_func = cls()
        opt_prob = Optimization(name=cls.name, opt_func=opt_func)

        opt_prob.add_obj('f')
        for i in range(len(cls.x_min)):
            opt_prob.add_var('x' + str(i + 1), lb=cls.x_min[i], ub=cls.x_max[i])
        for i in range(cls.n_eq_cons):
            opt_prob.add_econ('g' + str(i + 1))
        for i in range(cls.n_eq_cons, cls.n_cons):
            opt_prob.add_icon('g' + str(i + 1))

        opt_f, opt_x = opt_prob.solve(self.optimizer)

        # Check the solution

        self.assertLessEqual(np.linalg.norm(opt_x - cls.opt_x), dtol)

        if atol is None:
            self.assertLessEqual(abs(opt_f - cls.opt_f),
                                 rtol*abs(cls.opt_f))
        else:
            self.assertLessEqual(abs(opt_f - cls.opt_f), atol)

    def test_rastrigin(self):
        self._setup_test(Rastrigin, atol=0.002, dtol=0.001)

    def test_rosenbrock(self):
        self._setup_test(Rosenbrock, atol=0.002)

    def test_eggholder(self):
        self._setup_test(EggHolder, swarm_size=150)

    def test_tp08(self):
        self._setup_test(TP08, dtol=0.1)

    def test_tp14(self):
        self._setup_test(TP14)

    def test_tp32(self):
        self._setup_test(TP32)

    def test_tp37(self):
        self._setup_test(TP37, dtol=0.1)

    def test_tp43(self):
        self._setup_test(TP43, dtol=0.1)


if __name__ == "__main__":
    unittest.main()
