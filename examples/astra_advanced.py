#!/usr/bin/python
"""
This is an advanced example showing how to optimize the bunch length
as well as how to set equality and inequality constraints in with a
global optimizer.

Result after 15 outer iterations and 1880 function evaluations

St 64.916 fs

emitx 0.28011
gamma 22.242
max_Sx 1.3777

laser_spot 0.0400
main_sole_b 0.2674
gun_phase -10.0000
tws_phase -90.0000

Author: Jun Zhu
"""
from liso import Linac, LinacOptimization, ALPSO


# set up a linac
linac = Linac()

# add a beamline
#
# The first argument is the code name
# name: beamline name
# fin: simulation input file path
# template: simulation input template file path.
# pout: output file name. It is assumed and must be in the same folder as 'fin'.
# timeout: used to deal with the bug in parallel-astra, the code may be stuck at
#          with certain parameter. So, please ensure timeout is longer than the
#          time required for one simulation.
linac.add_beamline('astra',
                   name='gun',
                   fin='astra_injector/injector.in',
                   template='astra_advanced/injector.in.000',
                   pout='injector.0450.001',
                   timeout=90)

# set the optimizer
optimizer = ALPSO()
optimizer.swarm_size = 40

# set an optimization problem
opt = LinacOptimization(linac)

# There are two options to access a parameter of a linac:
# 1. Use a string: the string must have the form
#        beamline_name.WatchParameters_name.param_name
#        or
#        beamline_name.LineParameters_name.param_name.
# 2. Use a function object: the function has only one argument which is the linac instance.

# objective: its value is the horizontal emittance at the end of the 'gun' beamline.
opt.add_obj('St', expr='gun.out.St', scale=1e15)

# equality constraint: its value is the No. of particles at the end of the 'gun' beamline
opt.add_econ('n_pars', expr='gun.out.n', eq=2000)
# inequality constraint with upper boundary (ub): its value is the beta (x) function at the end of the 'gun' beamline.
opt.add_icon('emitx', expr='gun.out.emitx', scale=1e6,  ub=0.3)
# inequality constraint with lower boundary (lb): its value is the Lorentz factor at the end of the 'gun' beamline.
opt.add_icon('gamma', func=lambda a: a.gun.out.gamma,  lb=20.0)
# inequality constraint with upper boundary (ub): its value is the maximum beam size (x) throughout the 'gun' beamline.
opt.add_icon('max_Sx', func=lambda a: a.gun.max.Sx*1e3, ub=3.0)

# variables with lower boundary (lb) and upper boundary (ub)
opt.add_var('laser_spot',  value=0.1, lb=0.04, ub=0.5)
opt.add_var('main_sole_b', value=0.2, lb=0.00, ub=0.4)
opt.add_var('gun_phase', value=0.0, lb=-10, ub=10)
opt.add_var('tws_phase', value=0.0, lb=-90, ub=0)

opt.workers = 12  # use parallel Astra
opt.printout = 1  # print the optimization process
opt.solve(optimizer)  # run the optimization

