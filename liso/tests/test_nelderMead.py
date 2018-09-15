"""
Unittest of Nelder-Mead optimizer.

Author: Jun Zhu, zhujun981661@gmail.com
"""
import unittest

import numpy as np

from liso import Optimization, NelderMead
from liso.optimizers.nelder_mead import (
    _compute_centroid, _compute_reflection, _compute_expansion, _compute_shrink
)
from liso.exceptions import OptimizationConstraintSupportError

from .opt_problems import TP08, Rastrigin, Rosenbrock, EggHolder


class TestNelderMead(unittest.TestCase):
    def setUp(self):
        self.optimizer = NelderMead()

        self.simplex3 = [(np.array([0.0, 0.0]), 0.0),
                         (np.array([1.0, 0.0]), 1.0),
                         (np.array([0.0, 1.0]), 2.0)]
        self.xc3 = np.array([0.5,  0.0])
        self.xr3 = np.array([1.0, -1.0])
        self.xe3 = np.array([1.5, -2.0])
        self.shrink3 = [(np.array([0.0, 0.0]), 0.0),
                        (np.array([0.5, 0.0]), 0.5),
                        (np.array([0.0, 0.5]), 1.0)]

    def test_compute_centroid(self):
        xc = _compute_centroid(self.simplex3)
        self.assertAlmostEqual(xc[0], self.xc3[0], 1e-7)
        self.assertAlmostEqual(xc[1], self.xc3[1], 1e-7)

    def test_compute_reflection(self):
        xr = _compute_reflection(self.simplex3[-1][0], self.xc3, 1.0)
        self.assertAlmostEqual(xr[0], self.xr3[0], 1e-7)
        self.assertAlmostEqual(xr[1], self.xr3[1], 1e-7)

    def test_compute_expansion(self):
        xe = _compute_expansion(self.xr3, self.xc3, 2.0)
        self.assertAlmostEqual(xe[0], self.xe3[0], 1e-7)
        self.assertAlmostEqual(xe[1], self.xe3[1], 1e-7)

    def test_compute_shrink(self):
        vertices = _compute_shrink(self.simplex3, 0.5)
        self.assertAlmostEqual(vertices[0][0], self.shrink3[0][0][0], 1e-7)
        self.assertAlmostEqual(vertices[0][1], self.shrink3[0][0][1], 1e-7)
        self.assertAlmostEqual(vertices[1][0], self.shrink3[1][0][0], 1e-7)
        self.assertAlmostEqual(vertices[1][1], self.shrink3[1][0][1], 1e-7)
        self.assertAlmostEqual(vertices[2][0], self.shrink3[2][0][0], 1e-7)
        self.assertAlmostEqual(vertices[2][1], self.shrink3[2][0][1], 1e-7)

    def _setup_test(self, cls, x_init, *, atol=None, rtol=1e-3, dtol=1e-4):
        """Set up a test.

        :param cls: OptimizationTestProblem instance
            A test problem.
        :param x_init: list
            Initial x value.
        :param atol: float
            Absolute tolerance of the objective.
        :param rtol: float
            Relative tolerance of the objective.
        :param dtol: float
            Absolute tolerance of the position (L2 norm).
        """
        opt_prob = Optimization(cls.name, opt_func=cls())

        opt_prob.add_obj('f')
        for i in range(len(cls.x_min)):
            opt_prob.add_var('x' + str(i + 1), value=x_init[i],
                             lb=cls.x_min[i], ub=cls.x_max[i])
        for i in range(cls.n_eq_cons):
            opt_prob.add_econ('g' + str(i + 1))
        for i in range(cls.n_eq_cons, cls.n_cons):
            opt_prob.add_icon('g' + str(i + 1))

        opt_f, opt_x = opt_prob.solve(self.optimizer)

        # Check the solution
        self.assertLessEqual(np.linalg.norm(opt_x - cls.opt_x), dtol)

        if atol is None:
            self.assertLessEqual(abs(opt_f - cls.opt_f), rtol*abs(cls.opt_f))
        else:
            self.assertLessEqual(abs(opt_f - cls.opt_f), atol)

    def test_raise(self):
        with self.assertRaises(OptimizationConstraintSupportError):
            self.assertRaises(self._setup_test(TP08, [10, 10]))

    def test_rastrigin(self):
        self._setup_test(Rastrigin, [0.1, 0.1], atol=0.002, dtol=0.001)

    def test_rosenbrock(self):
        self._setup_test(Rosenbrock, [-1, -1], atol=0.002, dtol=0.01)

    def test_eggholder(self):
        self._setup_test(EggHolder, [500, 400], rtol=0.01, dtol=0.2)


if __name__ == "__main__":
    unittest.main()
