#!/usr/bin/python
"""
This is a basic example showing how to study jitter.
"""
from liso import Linac, LinacJitter

#######################################################################
# setup the optimization problem
#######################################################################

# Instantiate the optimization
linac = Linac()
linac.add_beamline('astra',
                   name='gun',
                   fin='astra_basic/injector.in',
                   template='astra_jitter/injector.in.000',
                   pout='injector.0400.001')

print(linac)

jt = LinacJitter(linac)

jt.add_response('emitx', expr='gun.out.emitx', scale=1e6)  # response
jt.add_response('Ct', expr='gun.out.Ct', scale=1e15)  # response
jt.add_response('gamma', expr='gun.out.gamma')  # response
jt.add_jitter('gun_gradient', value=110, sigma=-0.001)  # jitter
jt.add_jitter('gun_phase', value=0.0, sigma=0.01)  # jitter
jt.add_jitter('tws_gradient', value=30, sigma=-0.001)  # jitter
jt.add_jitter('tws_phase', value=0.0, sigma=0.01)  # jitter

jt._DEBUG = True
jt.run(10)
