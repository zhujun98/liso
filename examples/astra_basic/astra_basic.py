"""
This is a basic example showing how to optimize the emittance in ASTRA
with a local search optimizer.

```
python astra_basic.py --workers <number of cpu cores>
```

The solution of running the following code with space-charge effect is
    emitx_um    = 0.3195 
at
    laser_spot  = 0.1189
    main_sole_b = 0.2187


Author: Jun Zhu
"""
from liso import Linac, LinacOptimization, NelderMead
from liso.logging import logger
logger.setLevel('DEBUG')


linac = Linac()  # instantiate a Linac

# add a beamline
#
# The first argument is the code name
# name: beamline name
# swd: simulation working directory
# fin: simulation input file name
# template: simulation input template file path.
# pout: output file name. It must be in the same folder as 'fin'.
linac.add_beamline('astra',
                   name='gun',
                   swd='../astra_files',
                   fin='injector.in',
                   template='injector.in.000',
                   pout='injector.0450.001')

mapping = {
    'laser_spot': 0.1,
    'main_sole_b': 0.2,
}

opt = LinacOptimization(linac)  # instantiate Optimization (problem)

# There are two options to access a parameter of a linac:
# 1. Use a string: the string must have the form
#        beamline_name.WatchParameters_name.param_name
#    or
#        beamline_name.LineParameters_name.param_name.
# 2. Use a function object: the function has only one argument which is
#    the linac instance.

# add the objective (the horizontal emittance at the end of the 'gun' beamline)
opt.add_obj('emitx_um', expr='gun.out.emitx', scale=1e6)

# add variables with lower boundary (lb) and upper boundary (ub)
opt.add_var('laser_spot',  value=0.10, lb=0.04, ub=0.3)
opt.add_var('main_sole_b', value=0.20, lb=0.00, ub=0.4)

optimizer = NelderMead()  # instantiate an optimizer
opt.solve(optimizer)  # run the optimization
