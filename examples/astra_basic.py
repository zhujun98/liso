#!/usr/bin/python
"""
This is a basic example showing how to optimize the emittance in ASTRA
with a local search optimizer.

The solution is 0.1543 at laser_spot = 0.040 and main_sole_b = 0.2750.
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

print(linac)

# set the optimizer
if USE_PYOPT:
    optimizer = SDPEN()
    optimizer.setOption('alfa_stop', 1e-2)
else:
    optimizer = ALPSO()

opt = LinacOptimization(linac)

opt.add_obj('emitx_um', expr='gun.out.emitx', scale=1.e6)  # objective
opt.add_icon('g1', func=lambda a: a.gun.out.St*1e12 - 3)  # inequality constraint
opt.add_icon('g2', func=lambda a: a.gun.max.Sx*1e3 - 0.1)  # inequality constraint
opt.add_var('laser_spot', value=0.1, lb=0.04, ub=0.3)  # variable
opt.add_var('main_sole_b', value=0.1, lb=0.0, ub=0.4)  # variable

opt.workers = 2
opt._DEBUG = True
opt.solve(optimizer)  # Run the optimization
