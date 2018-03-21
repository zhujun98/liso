#!/usr/bin/python

"""
This is a basic example showing how to optimize the beam size
in IMPACT-T with a local search optimizer.

It should end up with Sx = 0.0486 mm, Sy = 0.2998 mm.
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
linac.add_beamline('impactt',
                   name='matching',
                   fin='impactt_basic/ImpactT.in',
                   template='impactt_basic/ImpactT.in.000',
                   pout='fort.107',
                   charge=0.0)

print(linac)

# set the optimizer
if USE_PYOPT:
    optimizer = SDPEN()
    optimizer.setOption('alfa_stop', 1e-2)
else:
    optimizer = ALPSO()


def obj_func(linac):
    """Define objective and constraint functions."""
    # define objective
    f = linac['matching'].out.Sx*1.e3

    # define constraint
    g = list()

    # After initializing the Beamline Object, it will automatically add
    # an attribute which is a Watch object (name='out'). 'Watch' is an
    # abstraction for a location where the code dumps particle file, one
    # can retrieve the beam parameters at this location by, for instance,
    # linac[beamline name].out.Sx (rms beam size),
    # linac[beamline name].out.emitx (horizontal emittance).
    g.append(linac['matching'].out.Sy*1.e3 - 0.15)
    # After initializing the Beamline Object, it will automatically add
    # an attribute which is a Line object (name='all'). 'Line' is an
    # abstraction for beam statistic along the whole or a section of the
    # beamline, one can retrieve the beam statistics by, for instance,
    # linac[beamline name].all.Sx.max (max rms beam size),
    # linac[beamline name].all.betax.ave (average beta function).
    g.append(linac['matching'].all.Sx.max*1.e3 - 0.20)

    print(f, g)
    return f, g


opt = LinacOptimization(linac, obj_func)

opt.add_obj('Sx')  # objective
opt.add_icon('g1')  # inequality constraint
opt.add_icon('g2')  # inequality constraint
opt.add_var('MQZM1_G', value=0.0, lower=-12.0, upper=12.0)  # variable
opt.add_var('MQZM2_G', value=0.0, lower=-12.0, upper=12.0)  # variable

opt.workers = 2
opt.solve(optimizer)  # Run the optimization
