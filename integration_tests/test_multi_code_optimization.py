#!/usr/bin/python
"""
Unittest of optimization with two simulation codes for two different
parts of a linac.

Author: Jun Zhu, zhujun981661@gmail.com
"""
import os
import glob
import unittest

from liso import Linac, LinacOptimization, ALPSO
from liso.logging import create_logger

logger = create_logger(__name__)

test_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'multi_code_optimization'
))


class TestMultiCodeOptimization(unittest.TestCase):
    def setUp(self):
        linac = Linac()

        linac.add_beamline(
            'astra',
            name='gun',
            fin=os.path.join(test_path, 'astra/injector.in'),
            template=os.path.join(test_path, 'astra/injector.in.000'),
            pout='injector.0100.001')

        linac.add_beamline(
            'impactt',
            name='matching',
            fin=os.path.join(test_path, 'impactt/ImpactT.in'),
            template=os.path.join(test_path, 'impactt/ImpactT.in.000'),
            pout='fort.106',
            charge=10e-12)

        self.opt = LinacOptimization(linac)

        self.opt.add_obj('emitx_um', expr='matching.out.emitx', scale=1.0e6)
        self.opt.add_icon('g1', func=lambda a: max(a.gun.max.Sx, a.matching.max.Sx), scale=1.0e3, ub=1.5)
        self.opt.add_icon('g2', func=lambda a: a.matching.out.betax, lb=40)
        self.opt.add_icon('g3', func=lambda a: a.matching.out.betax, ub=60)
        self.opt.add_icon('g4', func=lambda a: a.matching.out.betay, ub=20)

        self.opt.add_var('laser_spot', value=0.1, lb=0.04, ub=0.50)
        self.opt.add_var('main_sole_b', value=0.1, lb=0.00, ub=0.40)
        self.opt.add_var('MQZM1_G', value=0.0, lb=-6.0, ub=6.0)
        self.opt.add_var('MQZM2_G', value=0.0, lb=-6.0, ub=6.0)

    def tearDown(self):
        for file in glob.glob(os.path.join(test_path, "astra/injector.*.001")):
            os.remove(file)
        os.remove(os.path.join(test_path, "astra/injector.in"))
        for file in glob.glob(os.path.join(test_path, "impactt/fort.*")):
            os.remove(file)
        os.remove(os.path.join(test_path, "impactt/ImpactT.in"))

    def test_not_raise(self):
        logger.info("\n - Test multi-code optimization with ALPSO! - \n")

        optimizer = ALPSO()
        optimizer.swarm_size = 20
        optimizer.max_inner_iter = 3
        optimizer.min_inner_iter = 1
        optimizer.max_outer_iter = 3

        self.opt.monitor_time = True
        self.opt.solve(optimizer)


if __name__ == "__main__":
    unittest.main()
