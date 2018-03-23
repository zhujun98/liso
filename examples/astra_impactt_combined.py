#!/usr/bin/python
"""
This is an example showing how to simulate different part of a linac
using different code.
"""
from liso import Linac

USE_PYOPT = True

if USE_PYOPT:
    from pyOpt import SDPEN
    from liso import PyoptLinacOptimization as LinacOptimization
else:
    from liso import LinacOptimization
    from liso import ALPSO

#######################################################################
# setup the optimization problem
#######################################################################

# Instantiate the optimization
linac = Linac()

linac.add_beamline('astra',
                   name='gun',
                   fin='astra_basic/injector.in',
                   template='astra_basic/injector.in.000',
                   pout='injector.0400.001')

linac.add_beamline('impactt',
                   name='chicane',
                   fin='impactt_basic/ImpactT.in',
                   template='impactt_basic/ImpactT.in.000',
                   pout='fort.107',
                   charge=0.0001,
                   z0=0.0)

linac.add_watch('chicane', 'out1', 'fort.107', halo=0.1, tail=0.2)

print(linac)

# set the optimizer
if USE_PYOPT:
    optimizer = SDPEN()
    optimizer.setOption('alfa_stop', 1e-2)
else:
    optimizer = ALPSO()

opt = LinacOptimization(linac)

opt.add_obj('emitx_um', expr='chicane.out.emitx', scale=1.0e6)  # objective
opt.add_icon('g1', func=lambda a: a.chicane.out1.Sy*1e3, ub=0.06)  # inequality constraint
opt.add_icon('g2', func=lambda a: a.chicane.out1.Sx*1e3, ub=0.02)  # inequality constraint
opt.add_icon('g3', func=lambda a: max(a.gun.max.Sx, a.chicane.max.Sx)*1e3, ub=0.2)  # inequality constraint
opt.add_icon('g4', func=lambda a: max(a.gun.max.Sy, a.chicane.max.Sy)*1e3, ub=0.2)  # inequality constraint

opt.add_var('laser_spot', value=0.1, lb=0.04, ub=0.3)  # variable
opt.add_var('main_sole_b', value=0.1, lb=0.0, ub=0.4)  # variable
opt.add_var('MQZM1_G', value=0.0, lb=-2.0, ub=2.0)  # variable
opt.add_var('MQZM2_G', value=0.0, lb=-2.0, ub=2.0)  # variable

opt.workers = 1
opt._DEBUG = True
opt.solve(optimizer)  # Run the optimization
