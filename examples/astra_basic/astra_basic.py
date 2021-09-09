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

Note: Remember to modify RTD if the example file is modified.

Author: Jun Zhu
"""
from liso import Linac, LinacOptimization, NelderMead
from liso.logging import logger
logger.setLevel('DEBUG')


# Instantiate a Linac object
linac = Linac()

# Add a beamline which is simulated by ASTRA
linac.add_beamline('astra',
                   name='gun',
                   swd='../astra_files',
                   fin='injector.in',
                   template='injector.in.000',
                   pout='injector.0450.001')

# Define parameters which are defined within a pair of angle bracket in the
# template file.
params = {
    'laser_spot': 0.1,
    'main_sole_b': 0.2,
}

# Run the simulation (test whether everything is set up properly)
linac.run(params)

# Instantiate an Optimization object
opt = LinacOptimization(linac)

# There are two options to access a parameter of a linac:
# 1. Use a string: the string must have the form
#        beamline_name.WatchParameters_name.param_name
#    or
#        beamline_name.LineParameters_name.param_name.
# 2. Use a function object: the function has only one argument which is
#    the linac instance.

# Add the objective (the horizontal emittance at the end of the 'gun' beamline)
opt.add_obj('emitx_um', expr='gun.out.emitx', scale=1e6)

# Add variables with lower boundary (lb) and upper boundary (ub)
opt.add_var('laser_spot',  value=0.10, lb=0.04, ub=0.3)
opt.add_var('main_sole_b', value=0.20, lb=0.00, ub=0.4)

# Instantiate an optimizer
optimizer = NelderMead()

# Run the optimization
opt.solve(optimizer)
