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
                   template='astra_basic/injector.in.000')
linac.add_watch(beamline='gun', name='gun_out', pfile='injector.0400.001')
linac.add_line(beamline='gun', name='gun_all', rootname='injector')

linac.add_beamline('impactt',
                   name='chicane',
                   fin='impactt_basic/ImpactT.in',
                   template='impactt_basic/ImpactT.in.000',
                   charge=0.0)
linac.add_watch(beamline='chicane', name='chicane_out', pfile='fort.107')
linac.add_line(beamline='chicane', name='chicane_all', rootname='fort')

print(linac)


def obj_func(linac):
    """Define objective and constraint functions."""
    # define objective
    f = linac['gun'].gun_out.emitx*1.e6

    # define constraint
    g = list()
    g.append(linac['gun'].gun_out.emitx*1e-6 - 0.1)
    g.append(linac['chicane'].chicane_out.St*1e-12 - 10)

    print(f)
    return f, g


# set the optimizer
if USE_PYOPT:
    optimizer = SDPEN()
    optimizer.setOption('alfa_stop', 1e-2)
else:
    optimizer = ALPSO()

opt = LinacOptimization(linac, obj_func)

opt.add_obj('emitx_um')  # objective
opt.add_icon('g1')  # equality constraint
opt.add_icon('g2')  # inequality constraint
opt.add_var('laser_spot', value=0.1, lower=0.04, upper=0.3)  # variable
opt.add_var('main_sole_b', value=0.1, lower=0.0, upper=0.4)  # variable
opt.add_var('MQZM1_G', value=0.0, lower=-2.0, upper=2.0)  # variable
opt.add_var('MQZM2_G', value=0.0, lower=-2.0, upper=2.0)  # variable

opt.solve(optimizer, workers=1)  # Run the optimization
