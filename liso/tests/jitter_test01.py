#!/usr/bin/python
"""
Test jitter study using ASTRA.
"""
from liso import Linac, LinacJitter


linac = Linac()

linac.add_beamline('astra',
                   name='gun',
                   fin='astra_gun/injector.in',
                   template='jitter_test01/injector.in.000',
                   pout='injector.0150.001')

print(linac)

# set an jitter problem
jt = LinacJitter(linac)

jt.add_response('emitx', expr='gun.out.emitx', scale=1e6)
jt.add_response('Ct', expr='gun.out.Ct', scale=1e15)
jt.add_response('gamma', expr='gun.out.gamma')

jt.add_jitter('gun_gradient', value=110, sigma=-0.001)
jt.add_jitter('gun_phase', value=0.0, sigma=0.01)

jt.workers = 1
jt.verbose = True
jt.run(10)
