#!/usr/bin/python
"""
Unittest of local optimization of a linac with different optimizers.

Author: Jun Zhu, zhujun981661@gmail.com
"""
import unittest

from liso import Linac, NelderMead, SDPEN, LinacOptimization
from .test_utils import print_title


class TestLocalOptimizer(unittest.TestCase):
    def setUp(self):
        linac = Linac()
        linac.add_beamline('astra',
                           name='gun',
                           fin='liso/tests/astra_gun/injector.in',
                           template='liso/tests/local_optimizer_test/injector.in.000',
                           pout='injector.0150.001')

        self.opt = LinacOptimization(linac)

        self.opt.add_obj('emitx_um', expr='gun.out.emitx', scale=1.e6)
        self.opt.add_var('laser_spot', value=0.1, lb=0.04, ub=0.3)
        self.opt.add_var('main_sole_b', value=0.1, lb=0.0, ub=0.4)

    def test_nelderMead(self):
        print_title("Test local optimizer NelderMead with ASTRA!")
        optimizer = NelderMead()

        self.opt.monitor_time = True
        self.opt.solve(optimizer)

    def test_sdpen(self):
        print_title("Test local optimizer SDPEN with ASTRA!")

        optimizer = SDPEN()
        optimizer.rtol = 1e-3

        self.opt.monitor_time = True
        self.opt.solve(optimizer)


if __name__ == "__main__":
    unittest.main()
