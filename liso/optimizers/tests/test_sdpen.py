"""
Unittest of SDPEN optimizer

Author: Jun Zhu, zhujun981661@gmail.com
"""
import unittest
import numpy as np

from liso.exceptions import OptimizationConstraintSupportError

from .opt_problems import Rastrigin, Rosenbrock, EggHolder, TP08, TP14, TP37

SKIP_TEST = False
try:
    from liso import Optimization, SDPEN
except ImportError:
    SKIP_TEST = True


@unittest.skipIf(SKIP_TEST is True, "Failed to import library")
class TestSDPEN(unittest.TestCase):
    def setUp(self):
        self.optimizer = SDPEN()

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
        SDPEN._constraint_on = True
        # pyOpt.SDPEN does not support equality constraint.
        with self.assertRaises(OptimizationConstraintSupportError):
            self.assertRaises(self._setup_test(TP14, [0., 0.]))

        SDPEN._constraint_on = False
        # inequality constraint is also turned off.
        with self.assertRaises(OptimizationConstraintSupportError):
            self.assertRaises(self._setup_test(TP08, [2., 2.]))

    def test_rastrigin(self):
        self._setup_test(Rastrigin, [0.1, 0.1], atol=0.002, dtol=0.001)

    def test_rosenbrock(self):
        self._setup_test(Rosenbrock, [-1, -1], atol=0.01, dtol=0.2)

    def test_eggholder(self):
        self._setup_test(EggHolder, [500, 400], rtol=0.01, dtol=0.2)

    # def test_tp08(self):
    #     self._setup_test(TP08, [15.8, 1.58], rtol=0.01, dtol=0.5)

    # def test_tp37(self):
    #     self._setup_test(TP37, [20, 20, 20], rtol=0.01, dtol=0.5)


if __name__ == "__main__":
    unittest.main()
