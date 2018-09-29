"""
This is an example showing how to simulate different part of a linac
using different code.

```
python astra_impactt_combined.py --workers <number of cpu cores>
```

This may not be a good example for optimization. It simply shows how
does a concatenated optimization work.
"""
import argparse

from liso import Linac, LinacOptimization, NelderMead


parser = argparse.ArgumentParser(description='Resnet benchmark')
parser.add_argument('--workers',
                    type=int,
                    nargs='?',
                    default='1',
                    help="Number of workers.")

args = parser.parse_args()

# ---------------------------------------------------------------------

linac = Linac()

# The first beamline is run by parallel ASTRA but the second one is run
# by series IMPACT-T

linac.add_beamline('astra',
                   name='gun',
                   fin='astra_injector/injector.in',
                   template='astra_impactt_combined/injector.in.000',
                   pout='injector.0450.001',
                   workers=args.workers)

# Add the second beamline
#
# Note: 'charge' is required for an ImpactT simulation but this argument
# will be ignored if the code is Astra.
linac.add_beamline('impactt',
                   name='chicane',
                   fin='impactt_lattice/ImpactT.in',
                   template='astra_impactt_combined/ImpactT.in.000',
                   pout='fort.106',
                   charge=1e-15)

opt = LinacOptimization(linac)

opt.add_obj('St_betaxy', func=lambda a: max(a.chicane.out.emitx*1e6,
                                            a.chicane.out.betax,
                                            a.chicane.out.betay))

opt.add_var('laser_spot',  value=0.1, lb=0.04, ub=0.30)
opt.add_var('main_sole_b', value=0.1, lb=0.00, ub=0.40)
opt.add_var('MQZM1_G', value=0.0, lb=-10, ub=10)
opt.add_var('MQZM3_G', value=0.0, lb=-10, ub=10)

opt.add_covar('MQZM2_G', 'MQZM1_G', scale=-1)
opt.add_covar('MQZM4_G', 'MQZM3_G', scale=-1)

opt.printout = 1

optimizer = NelderMead()
opt.solve(optimizer)
