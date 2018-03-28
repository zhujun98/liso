#!/usr/bin/python
"""
Wild ranges of gun gradient and phase are used in this test in order to
make many failed simulations. The API should be able to deal with these
cases.
"""
from liso import Linac, LinacOptimization, ALPSO


linac = Linac()
linac.add_beamline('astra',
                   name='gun',
                   fin='astra_gun/injector.in',
                   template='global_optimizer_test01/injector.in.000',
                   pout='injector.0150.001')

print(linac)

optimizer = ALPSO()
optimizer.swarm_size = 20
optimizer.max_inner_iter = 3
optimizer.min_inner_iter = 1
optimizer.max_outer_iter = 10

opt = LinacOptimization(linac)

opt.add_obj('emitx_um', expr='gun.out.emitx', scale=1.e6)
opt.add_icon('g1', func=lambda a: a.gun.max.Sx*1e3, ub=0.2)
opt.add_econ('g2', func=lambda a: a.gun.out.gamma, eq=10.0)

opt.add_var('laser_spot', value=0.1, lb=0.04, ub=0.3)
opt.add_var('main_sole_b', value=0.1, lb=0.0, ub=0.4)
opt.add_var('gun_gradient', value=130, lb=0.0, ub=130.0)
opt.add_var('gun_phase', value=0.0, lb=-90.0, ub=90.0)

opt.workers = 1
opt.verbose = True
opt.solve(optimizer)
