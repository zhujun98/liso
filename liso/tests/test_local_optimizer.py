#!/usr/bin/python
"""
Wild ranges of gun gradient and phase are used in this test in order to
make many failed simulations. The API should be able to deal with these
cases.
"""
import unittest

from liso import Linac, NelderMead, SDPEN, LinacOptimization


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
        optimizer = NelderMead()

        self.opt.monitor_time = True
        self.opt.solve(optimizer)

    def test_sdpen(self):
        optimizer = SDPEN()
        optimizer.rtol = 1e-2

        self.opt.monitor_time = True
        self.opt.solve(optimizer)
