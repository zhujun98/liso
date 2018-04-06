#!/usr/bin/python
"""
This is an example showing how to simulate different part of a linac
using different code.

TODO: under development
"""
from liso import Linac, LinacOptimization, ALPSO


# Set up the linac
linac = Linac()

# Add the first beamline
linac.add_beamline('astra',
                   name='gun',
                   fin='astra_impactt_combined/astra/injector.in',
                   template='astra_impactt_combined/astra/injector.in.000',
                   pout='injector.0400.001')

# Add the second beamline
#
# Note: 'charge' is required for an ImpactT simulation but this argument
# will be ignored if the code is Astra.
linac.add_beamline('impactt',
                   name='chicane',
                   fin='astra_impactt_combined/impactt/ImpactT.in',
                   template='astra_impactt_combined/impactt/ImpactT.in.000',
                   pout='fort.107',
                   charge=1e-12)

print(linac)

opt = LinacOptimization(linac)

opt.add_obj('St', expr='chicane.out.St', scale=1.0e15)  # objective
opt.add_icon('g3', func=lambda a: a.matching.out.emitx*1e6, ub=0.2)  # inequality constraint

opt.add_var('laser_spot',  value=0.1, lb=0.04, ub=0.50)  # variable
opt.add_var('main_sole_b', value=0.1, lb=0.00, ub=0.40)  # variable
opt.add_var('tws_phase', value=0.1, lb=0.00, ub=0.40)  # variable
opt.add_var('dipole_by', value=0.0, lb=0.00, ub=0.40)  # variable

opt.workers = 2
opt.printout = 1

optimizer = ALPSO()
opt.solve(optimizer)
