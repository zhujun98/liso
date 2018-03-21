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


def obj_func(linac):
    """Define objective and constraint functions."""
    # define objective
    f = linac['gun'].out.emitx*1.e6

    # define constraint
    g = list()

    # After initializing the Beamline Object, it will automatically add
    # an attribute which is a Watch object (name='out'). 'Watch' is an
    # abstraction for a location where the code dumps particle file, one
    # can retrieve the beam parameters at this location by, for instance,
    # linac[beamline name].out.Sx (rms beam size),
    # linac[beamline name].out.emitx (horizontal emittance).
    g.append(linac['gun'].out.St - 5e-12)
    # After initializing the Beamline Object, it will automatically add
    # an attribute which is a Line object (name='all'). 'Line' is an
    # abstraction for beam statistic along the whole or a section of the
    # beamline, one can retrieve the beam statistics by, for instance,
    # linac[beamline name].all.Sx.max (max rms beam size),
    # linac[beamline name].all.betax.ave (average beta function).
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
opt.add_var('laser_spot', value=0.1, lb=0.04, ub=0.3)  # variable
opt.add_var('main_sole_b', value=0.1, lb=0.0, ub=0.4)  # variable

opt.workers = 2
opt.solve(optimizer)  # Run the optimization
