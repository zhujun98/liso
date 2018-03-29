#!/usr/bin/python
"""
Wild ranges of gun gradient and phase are used in this test in order to
make many failed simulations. The API should be able to deal with these
cases.
"""
import unittest

from liso import Linac, ALPSO, LinacOptimization


class TestGlobalOptimizer(unittest.TestCase):
    def setUp(self):
        linac = Linac()
        linac.add_beamline('astra',
                           name='gun',
                           fin='liso/tests/astra_gun/injector.in',
                           template='liso/tests/global_optimizer_test01/injector.in.000',
                           pout='injector.0150.001')

        print(linac)

        self.opt = LinacOptimization(linac)

        self.opt.add_obj('emitx_um', expr='gun.out.emitx', scale=1.e6)
        self.opt.add_icon('g1', func=lambda a: a.gun.max.Sx * 1e3, ub=0.2)
        self.opt.add_econ('g2', func=lambda a: a.gun.out.gamma, eq=10.0)

        self.opt.add_var('laser_spot', value=0.1, lb=0.04, ub=0.3)
        self.opt.add_var('main_sole_b', value=0.1, lb=0.0, ub=0.4)
        self.opt.add_var('gun_gradient', value=130, lb=0.0, ub=130.0)
        self.opt.add_var('gun_phase', value=0.0, lb=-90.0, ub=0.0)

    def test_not_raise(self):
        optimizer = ALPSO()
        optimizer.swarm_size = 10
        optimizer.max_inner_iter = 3
        optimizer.min_inner_iter = 1
        optimizer.max_outer_iter = 5

        self.opt.workers = 1
        self.opt.time_monitor = True
        self.opt.solve(optimizer)
