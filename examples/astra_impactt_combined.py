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
                   template='astra_basic/injector.in.000',
                   pout='injector.0400.001')

linac.add_beamline('impactt',
                   name='chicane',
                   fin='impactt_basic/ImpactT.in',
                   template='impactt_basic/ImpactT.in.000',
                   pout='fort.107',
                   charge=0.0001,
                   z0=0.0)

linac.add_watch('gun', 'out1', 'injector.0400.001', halo=0.1)
linac.add_watch('chicane', 'out1', 'fort.107', tail=0.2)

print(linac)


def obj_func(linac):
    """Define objective and constraint functions."""
    # define objective
    f = (linac['chicane'].out.Sx + linac['chicane'].out.Sy)*1.e3

    # define constraint
    g = list()
    g.append(linac['gun'].out1.emitx*1e-6 - 0.1)
    g.append(linac['chicane'].out1.St*1e-12 - 10)

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
opt.add_var('laser_spot', value=0.1, lb=0.04, ub=0.3)  # variable
opt.add_var('main_sole_b', value=0.1, lb=0.0, ub=0.4)  # variable
opt.add_var('MQZM1_G', value=0.0, lb=-2.0, ub=2.0)  # variable
opt.add_var('MQZM2_G', value=0.0, lb=-2.0, ub=2.0)  # variable

opt.workers = 1
opt.solve(optimizer)  # Run the optimization
