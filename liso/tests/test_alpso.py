"""
Unittest for ALPSO

Eggholder and TP37 have higher probabilities of failure.
"""
import unittest
import numpy as np

from liso import Optimization, ALPSO
from .opt_test_problems import *


class TestALPSO(unittest.TestCase):
    def setUp(self):
        self.optimizer = ALPSO()

    def _setup_test(self, cls, swarm_size=100, *, atol=None, rtol=1e-3, dtol=1e-4, printout=1):
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
        :param printout: int
            Printout level of the optimizer.
        """
        self.optimizer.seed = 2  # Get consistent result
        self.optimizer.printout = printout
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

        opt_f, opt_x, _ = self.optimizer(opt_prob)

        # Check the solution

        # print(np.linalg.norm(opt_x - cls.opt_x))
        # if cls.opt_f != 0:
        #     print(abs(opt_f - cls.opt_f)/abs(cls.opt_f))
        # else:
        #     print(abs(opt_f - cls.opt_f))

        # x
        self.assertLessEqual(np.linalg.norm(opt_x - cls.opt_x), dtol)

        # f
        if atol is None:
            self.assertLessEqual(abs(opt_f - cls.opt_f),
                                 rtol*abs(cls.opt_f))
        else:
            self.assertLessEqual(abs(opt_f - cls.opt_f), atol)

    def test_rastrigin(self):
        self._setup_test(Rastrigin, 150, atol=0.002, dtol=0.001)

    def test_rosenbrock(self):
        self._setup_test(Rosenbrock, 150, atol=0.002, dtol=0.01)

    def test_eggholder(self):
        self._setup_test(EggHolder, 150, rtol=0.01, dtol=0.1)

    def test_tp08(self):
        self._setup_test(TP08, rtol=0.01, dtol=0.5)

    def test_tp14(self):
        self._setup_test(TP14, 150, rtol=0.02, dtol=0.01)

    def test_tp32(self):
        self._setup_test(TP32, rtol=0.01, dtol=0.01)

    def test_tp37(self):
        self._setup_test(TP37, 150, rtol=0.02, dtol=0.2)

    def test_tp43(self):
        self._setup_test(TP43, rtol=0.02, dtol=0.3)


if __name__ == "__main__":
    unittest.main()