#!/usr/bin/python
"""
This is a basic example showing how to optimize the beam size
in IMPACT-T with a local search optimizer.
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
    optimizer.swarm_size = 20
    optimizer.max_inner_iter = 3
    optimizer.min_inner_iter = 1
    optimizer.max_outer_iter = 10


def g2(a):
    """Define a constraint function.

    :param a: Linac
        A Linac instance.
    """
    return (a.matching.max.Sx + a.matching.max.Sy)*1.e3 - 0.4


opt = LinacOptimization(linac)

opt.add_obj('Sx', expr='matching.out.Sx', scale=1e3)  # objective
opt.add_econ('g1', func=lambda a: a.matching.out.Sy*1e3 - 0.1)  # equality constraint
opt.add_icon('g2', func=g2)  # inequality constraint
opt.add_var('MQZM1_G', value=0.0, lb=-12.0, ub=12.0)  # variable
opt.add_var('MQZM2_G', value=0.0, lb=-12.0, ub=12.0)  # variable

opt.workers = 2
opt._DEBUG = True
opt.solve(optimizer)  # Run the optimization
