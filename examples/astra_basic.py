#!/usr/bin/python
"""
This is a basic example showing how to optimize the emittance in ASTRA
with a local search optimizer.

The solution is 0.1543 at laser_spot = 0.040 and main_sole_b = 0.2750.
"""
from liso import Linac
from liso import PyoptLinacOptimization as LinacOptimization
from pyOpt import SDPEN


# set up a linac
linac = Linac()

# add a beamline
#
# The first argument is the code name
# name: beamline name
# fin: simulation input file path
# template: simulation input template file path.
# pout: output file name. It is assumed and must be in the same folder as 'fin'.
linac.add_beamline('astra',
                   name='gun',
                   fin='astra_injector/injector.in',
                   template='astra_basic/injector.in.000',
                   pout='injector.0450.001')

print(linac)

# set the optimizer
optimizer = SDPEN()
optimizer.setOption('alfa_stop', 1e-2)

# set an optimization problem
opt = LinacOptimization(linac)

# There are two options to access a parameter of a linac:
# 1. Use a string: the string must have the form
#        beamline_name.WatchParameters_name.param_name
#        or
#        beamline_name.LineParameters_name.param_name.
# 2. Use a function object: the function has only one argument which is the linac instance.

# objective: its value is the horizontal emittance at the end of the 'gun' beamline.
opt.add_obj('emitx_um', expr='gun.out.emitx', scale=1e6)
# inequality constraint with lower boundary (lb): its value is the beta (x) function at the end of the 'gun' beamline.
# opt.add_icon('g1', func=lambda a: a.gun.out.betax,  lb=10)
# inequality constraint with upper boundary (ub): its value is the maximum beam size (x) throughout the 'gun' beamline.
# opt.add_icon('g2', func=lambda a: a.gun.max.Sx*1e3, ub=2.0)

opt.add_var('laser_spot',  value=0.1, lb= 0.04, ub=0.5)  # variable with lower boundary (lb) and upper boundary (ub)
opt.add_var('main_sole_b', value=0.2, lb= 0.00, ub=0.4)  # variable

opt.workers = 2  # use parallel Astra
opt.verbose = True  # print the optimization process
opt.solve(optimizer)  # run the optimization
