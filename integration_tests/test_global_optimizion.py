#!/usr/bin/python
"""
Unittest of global optimization of a linac.

Wild ranges of gun gradient and phase are used in this test in order to
make many failed simulations. The API should be able to deal with these
cases.

Author: Jun Zhu, zhujun981661@gmail.com
"""
import os
import glob
import unittest

from liso import Linac, ALPSO, LinacOptimization

test_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'global_optimizer'
))


class TestGlobalOptimizer(unittest.TestCase):
    def setUp(self):
        linac = Linac()
        linac.add_beamline('astra',
                           name='gun',
                           fin=os.path.join(test_path, 'injector.in'),
                           template=os.path.join(test_path, 'injector.in.000'),
                           pout='injector.0150.001')

        self.opt = LinacOptimization(linac)
        # self.opt.printout = 1

        self.opt.add_obj('f', expr='gun.out.Sx', scale=1.e6)
        self.opt.add_icon('g1', func=lambda a: a.gun.max.emitx*1e6, ub=0.041)
        self.opt.add_econ('g2', func=lambda a: a.gun.out.gamma, eq=10.0)

        self.opt.add_var('laser_spot', value=0.1, lb=0.04, ub=0.3)
        self.opt.add_var('main_sole_b', value=0.1, lb=0.0, ub=0.3)
        self.opt.add_var('gun_gradient', value=130, lb=90.0, ub=130.0)

    def tearDown(self):
        for file in glob.glob(os.path.join(test_path, "injector.*.001")):
            os.remove(file)
        os.remove(os.path.join(test_path, "injector.in"))

    def test_not_raise(self):
        optimizer = ALPSO()

        opt_f, opt_x = self.opt.solve(optimizer)

        self.assertAlmostEqual(opt_f, 65.62, delta=0.10)
        self.assertAlmostEqual(opt_x[0], 0.040000, delta=0.00010)
        self.assertAlmostEqual(opt_x[1], 0.2521, delta=0.0010)
        self.assertAlmostEqual(opt_x[2], 100.4, delta=1.0)


if __name__ == "__main__":
    unittest.main()
