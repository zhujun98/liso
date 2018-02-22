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
linac.add_beamline(code='impactt',
                   name='matching',
                   input_file='impactt_basic/ImpactT.in',
                   template='impactt_basic/ImpactT.in.000')
linac.add_watch(beamline='matching', name='out', pfile='fort.107')
linac.add_line(beamline='matching', name='all', rootname='fort')

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
    g.append(linac['matching'].out.Sy*1.e3 - 0.15)

    print(f, g)
    return f, g


opt = LinacOptimization(linac, obj_func)

opt.add_obj('Sx')  # objective
opt.add_icon('g1')  # inequality constraint
opt.add_var('MQZM1_G', value=0.0, lower=-12.0, upper=12.0)  # variable
opt.add_var('MQZM2_G', value=0.0, lower=-12.0, upper=12.0)  # variable

opt.solve(optimizer, threads=2)  # Run the optimization
