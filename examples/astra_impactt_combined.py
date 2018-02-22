#!/usr/bin/python
"""
This is a basic example showing how to optimize the emittance in ASTRA
with a local search optimizer.

The solution is 0.1543 at laser_spot = 0.040 and main_sole_b = 0.2750.
"""
from linacopt import Linac
from linacopt import LinacOptimization
from linacopt import ALPSO

# from pyOpt import SDPEN
# from linacopt import PyoptLinacOptimization as LinacOptimization

#######################################################################
# setup the optimization problem
#######################################################################

# Instantiate the optimization
linac = Linac()
linac.add_beamline(code='astra',
                   name='gun',
                   input_file='astra_basic/injector.in',
                   template='astra_basic/injector.in.000')
linac.add_watch(beamline='gun', name='gun_out', pfile='injector.0400.001')
linac.add_line(beamline='gun', name='gun_all', rootname='injector')

linac.add_beamline(code='impactt',
                   name='chicane',
                   input_file='impactt_basic/ImpactT.in',
                   template='impactt_basic/ImpactT.in.000')
linac.add_watch(beamline='chicane', name='chicane_out', pfile='fort.107')
linac.add_line(beamline='chicane', name='chicane_all', rootname='fort')

print(linac)


def obj_func(linac):
    """Define objective and constraint functions."""
    # define objective
    f = linac['gun'].gun_out.emitx*1.e6

    # define constraint
    g = list()
    g.append(linac['gun'].gun_out.n)
    g.append(linac['chicane'].chicane_out.St*1e-12 - 10)

    print(f)
    return f, g


# set the optimizer
optimizer = ALPSO()

opt = LinacOptimization(linac, obj_func)

opt.add_obj('emitx_um')  # objective
opt.add_econ('g1')  # equality constraint
opt.add_icon('g2')  # inequality constraint
opt.add_var('laser_spot', value=0.1, lower=0.04, upper=0.3)  # variable
opt.add_var('main_sole_b', value=0.1, lower=0.0, upper=0.4)  # variable
opt.add_var('MQZM1_G', value=0.0, lower=-2.0, upper=2.0)  # variable
opt.add_var('MQZM2_G', value=0.0, lower=-2.0, upper=2.0)  # variable

opt.solve(optimizer, threads=1)  # Run the optimization
