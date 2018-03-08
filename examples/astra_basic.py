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
linac.add_beamline(code='astra',
                   name='gun',
                   input_file='astra_basic/injector.in',
                   template='astra_basic/injector.in.000')
linac.add_watch(beamline='gun', name='out', pfile='injector.0400.001')
linac.add_line(beamline='gun', name='all', rootname='injector')

print(linac)


def obj_func(linac):
    """Define objective and constraint functions."""
    # define objective
    f = linac['gun'].out.emitx*1.e6

    # define constraint
    g = list()
    g.append(linac['gun'].out.St - 5e-12)
    g.append(linac['gun'].all.Sx.max - 0.2e-3)

    print(f, g)
    return f, g


# set the optimizer
if USE_PYOPT:
    optimizer = SDPEN()
    optimizer.setOption('alfa_stop', 1e-2)
else:
    optimizer = ALPSO()

opt = LinacOptimization(linac, obj_func)

opt.add_obj('emitx_um')  # objective
opt.add_icon('g1')  # inequality constraint
opt.add_icon('g2')  # inequality constraint
opt.add_var('laser_spot', value=0.1, lower=0.04, upper=0.3)  # variable
opt.add_var('main_sole_b', value=0.1, lower=0.0, upper=0.4)  # variable

opt.solve(optimizer, workers=2)  # Run the optimization
