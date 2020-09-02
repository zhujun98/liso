"""
This is an example showing how to simulate different part of a linac
using different code.

```
python astra_impactt_combined.py --workers <number of cpu cores>
```

This may not be a good example for optimization. It simply shows how
does a concatenated optimization work.
"""
from liso import Linac, LinacOptimization, NelderMead
from liso.logging import logger
logger.setLevel('DEBUG')


linac = Linac()

# The first beamline is run by parallel ASTRA but the second one is run
# by series IMPACT-T

linac.add_beamline('astra',
                   name='gun',
                   swd='../astra_files',
                   fin='injector.in',
                   template='injector.in.000',
                   pout='injector.0450.001')

# Add the second beamline
#
# Note: 'charge' is required for an ImpactT simulation but this argument
# will be ignored if the code is Astra.
linac.add_beamline('impactt',
                   name='chicane',
                   swd='../impactt_files',
                   fin='ImpactT.in',
                   template='ImpactT.in.000',
                   pout='fort.106',
                   charge=1e-15)

opt = LinacOptimization(linac)

opt.add_obj('St_betaxy', func=lambda a: max(a['chicane'].out.emitx*1e6,
                                            a['chicane'].out.betax,
                                            a['chicane'].out.betay))

opt.add_var('gun.laser_spot',  value=0.1, lb=0.04, ub=0.30)
opt.add_var('gun.main_sole_b', value=0.1, lb=0.00, ub=0.40)
opt.add_var('chicane.MQZM1_G', value=0.0, lb=-10, ub=10)
opt.add_var('chicane.MQZM3_G', value=0.0, lb=-10, ub=10)

opt.add_covar('chicane.MQZM2_G', 'chicane.MQZM1_G', scale=-1)
opt.add_covar('chicane.MQZM4_G', 'chicane.MQZM3_G', scale=-1)

optimizer = NelderMead()
opt.solve(optimizer)
