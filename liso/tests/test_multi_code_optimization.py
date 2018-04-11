#!/usr/bin/python
"""
Unittest of optimization with two simulation codes for two different
parts of a linac.

Author: Jun Zhu
"""
import unittest

from liso import Linac, LinacOptimization, ALPSO
from .test_utils import print_title


class TestMultiCodeOptimization(unittest.TestCase):
    def setUp(self):
        linac = Linac()

        linac.add_beamline('astra',
                           name='gun',
                           fin='liso/tests/multi_code_optimization_test/astra/injector.in',
                           template='liso/tests/multi_code_optimization_test/astra/injector.in.000',
                           pout='injector.0100.001')

        linac.add_beamline('impactt',
                           name='matching',
                           fin='liso/tests/multi_code_optimization_test/impactt/ImpactT.in',
                           template='liso/tests/multi_code_optimization_test/impactt/ImpactT.in.000',
                           pout='fort.106',
                           charge=10e-12)

        print(linac)

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

    def test_not_raise(self):
        print_title("Test multi-code optimization with ALPSO!")

        optimizer = ALPSO()
        optimizer.swarm_size = 20
        optimizer.max_inner_iter = 3
        optimizer.min_inner_iter = 1
        optimizer.max_outer_iter = 3

        self.opt.monitor_time = True
        self.opt.solve(optimizer)


if __name__ == "__main__":
    unittest.main()
