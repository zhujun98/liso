"""
This is an example showing how to simulate different part of a linac
using different code.

This may not be a good example for optimization. It simply shows how
does a concatenated optimization work.
"""
from liso import Linac, LinacOptimization, NelderMead


linac = Linac()

linac.add_beamline('astra',
                   name='gun',
                   fin='astra_injector/injector.in',
                   template='astra_impactt_combined/injector.in.000',
                   pout='injector.0450.001')

# Add the second beamline
#
# Note: 'charge' is required for an ImpactT simulation but this argument
# will be ignored if the code is Astra.
linac.add_beamline('impactt',
                   name='chicane',
                   fin='impactt_lattice/ImpactT.in',
                   template='astra_impactt_combined/ImpactT.in.000',
                   pout='fort.106',
                   charge=1e-15)

opt = LinacOptimization(linac)

opt.add_obj('St_betaxy', func=lambda a: max(a.chicane.out.emitx*1e6,
                                            a.chicane.out.betax,
                                            a.chicane.out.betay))

opt.add_var('laser_spot',  value=0.1, lb=0.04, ub=0.30)
opt.add_var('main_sole_b', value=0.1, lb=0.00, ub=0.40)
opt.add_var('MQZM1_G', value=0.0, lb=-10, ub=10)
opt.add_var('MQZM3_G', value=0.0, lb=-10, ub=10)

opt.add_covar('MQZM2_G', 'MQZM1_G', scale=-1)
opt.add_covar('MQZM4_G', 'MQZM3_G', scale=-1)

opt.workers = 12
opt.printout = 1

optimizer = NelderMead()
opt.solve(optimizer)
