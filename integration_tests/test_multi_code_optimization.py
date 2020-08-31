"""
Unittest of optimization with two simulation codes for two different
parts of a linac.

Author: Jun Zhu, zhujun981661@gmail.com
"""
import os
import os.path as osp
import glob
import unittest

from liso import Linac, LinacOptimization, ALPSO

test_path = os.path.abspath(osp.join(
    os.path.dirname(__file__), 'multi_code_optimization'
))


class TestMultiCodeOptimization(unittest.TestCase):
    def setUp(self):
        linac = Linac()

        linac.add_beamline(
            'astra',
            name='gun',
            swd=osp.join(test_path, 'astra'),
            fin='injector.in',
            template=osp.join(test_path, 'astra/injector.in.000'),
            pout='injector.0100.001')

        linac.add_beamline(
            'impactt',
            name='matching',
            swd=osp.join(test_path, 'impactt'),
            fin='ImpactT.in',
            template=osp.join(test_path, 'impactt/ImpactT.in.000'),
            pout='fort.106',
            charge=10e-12)

        self.opt = LinacOptimization(linac)

        self.opt.add_obj('f', expr='matching.out.emitx', scale=1e6)

        self.opt.add_icon('g1', func=lambda a: a.gun.out.Sx * 1e3, ub=0.2)
        self.opt.add_icon('g2', func=lambda a: a.matching.out.betax, ub=0.2)
        self.opt.add_icon('g3', func=lambda a: a.matching.out.betay, ub=0.2)
        self.opt.add_icon(
            'g4',
            func=lambda a: abs(a.matching.out.betax - a.matching.out.betay),
            ub=0.01)

        self.opt.add_var('gun.laser_spot', value=0.1, lb=0.04, ub=0.30)
        self.opt.add_var('matching.MQZM1_G', value=0.0, lb=-6.0, ub=0.0)
        self.opt.add_var('matching.MQZM2_G', value=0.0, lb=0.0, ub=6.0)

    def tearDown(self):
        for file in glob.glob(os.path.join(test_path, "astra/injector.*.001")):
            os.remove(file)
        os.remove(os.path.join(test_path, "astra/injector.in"))
        for file in glob.glob(os.path.join(test_path, "impactt/fort.*")):
            os.remove(file)
        os.remove(os.path.join(test_path, "impactt/ImpactT.in"))

    def test_optimization(self):
        optimizer = ALPSO()

        opt_f, opt_x = self.opt.solve(optimizer)

        self.assertAlmostEqual(opt_f, 0.04025, delta=0.00040)
        self.assertAlmostEqual(opt_x[0], 0.04000, delta=0.00010)
        self.assertAlmostEqual(opt_x[1], -0.78, delta=0.030)
        self.assertAlmostEqual(opt_x[2], 0.96, delta=0.040)


if __name__ == "__main__":
    unittest.main()
