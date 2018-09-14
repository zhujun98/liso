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


linac = Linac()

linac.add_beamline('astra',
                   name='gun',
                   fin='astra_injector/injector.in',
                   template='astra_advanced/injector.in.000',
                   pout='injector.0450.001',
                   timeout=90)

opt = LinacOptimization(linac)

opt.add_obj('St', expr='gun.out.St', scale=1e15)

# equality constraint (the No. of particles at the end of the 'gun' beamline)
opt.add_econ('n_pars', expr='gun.out.n', eq=2000)
# inequality constraint (the beta [x] function at the end of the 'gun' beamline) with the upper boundary.
opt.add_icon('emitx', expr='gun.out.emitx', scale=1e6,  ub=0.3)
# inequality constraint (the Lorentz factor at the end of the 'gun' beamline) with the lower boundary.
opt.add_icon('gamma', func=lambda a: a.gun.out.gamma,  lb=20.0)
# inequality constraint (the maximum beam size [x] throughout the 'gun' beamline) with upper boundary.
opt.add_icon('max_Sx', func=lambda a: a.gun.max.Sx*1e3, ub=3.0)

opt.add_var('laser_spot',  value=0.1, lb=0.04, ub=0.5)
opt.add_var('main_sole_b', value=0.2, lb=0.00, ub=0.4)
opt.add_var('gun_phase', value=0.0, lb=-10, ub=10)
opt.add_var('tws_phase', value=0.0, lb=-90, ub=0)

opt.workers = 12
opt.printout = 1

optimizer = ALPSO()
optimizer.swarm_size = 40  # configure the optimizer
opt.solve(optimizer)
