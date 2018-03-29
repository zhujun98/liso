#!/usr/bin/python
"""
Test optimization with two simulation codes for two different parts
of a linac.
"""
from liso import Linac, PyoptLinacOptimization
from pyOpt import SDPEN


linac = Linac()

linac.add_beamline('astra',
                   name='gun',
                   fin='multi_code_optimization_test01/astra/injector.in',
                   template='multi_code_optimization_test01/astra/injector.in.000',
                   pout='injector.0100.001')

linac.add_beamline('impactt',
                   name='matching',
                   fin='multi_code_optimization_test01/impactt/ImpactT.in',
                   template='multi_code_optimization_test01/impactt/ImpactT.in.000',
                   pout='fort.106',
                   charge=10e-12)

print(linac)

opt = PyoptLinacOptimization(linac)

opt.add_obj('emitx_um', expr='matching.out.emitx', scale=1.0e6)
opt.add_icon('g1', func=lambda a: max(a.gun.max.Sx, a.matching.max.Sx), scale=1.0e3, ub=1.5)
opt.add_icon('g2', func=lambda a: a.matching.out.betax, lb=40)
opt.add_icon('g3', func=lambda a: a.matching.out.betax, ub=60)
opt.add_icon('g4', func=lambda a: a.matching.out.betay, ub=20)

opt.add_var('laser_spot',  value=0.1, lb=0.04, ub=0.50)
opt.add_var('main_sole_b', value=0.1, lb=0.00, ub=0.40)
opt.add_var('MQZM1_G', value=0.0, lb=-6.0, ub=6.0)
opt.add_var('MQZM2_G', value=0.0, lb=-6.0, ub=6.0)

opt.verbose = True

optimizer = SDPEN()
optimizer.setOption('alfa_stop', 1e-2)

opt.solve(optimizer)
