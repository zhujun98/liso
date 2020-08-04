"""
This is a basic example showing how to study jitter.
"""
from liso import Linac, LinacJitter


linac = Linac()  # instantiate a Linac

linac.add_beamline('astra',
                   name='gun',
                   fin='astra_injector/injector.in',
                   template='astra_jitter/injector.in.000',
                   pout='injector.0450.001')

jt = LinacJitter(linac)  # instantiate a Jitter (problem)

jt.add_response('emitx', expr='gun.out.emitx', scale=1e6)  # response, which is the horizontal emittance in micro-meter
jt.add_response('Ct', expr='gun.out.Ct', scale=1e15)  # response, which is the timing in fs
jt.add_response('gamma', expr='gun.out.gamma')  # response, which is the Lorentz factor

jt.add_jitter('gun_gradient', value=130, sigma=-0.001)  # nominal value = 110, standard deviation = 110 * 0.001
jt.add_jitter('gun_phase', value=0.0, sigma=0.01)  # nominal value = 0.0, standard deviation = 0.01
jt.add_jitter('tws_gradient', value=30, sigma=-0.001)  # nominal value = 30, standard deviation = 30 * 0.001
jt.add_jitter('tws_phase', value=-20.0, sigma=0.01)  # nominal value = 0.0, standard deviation = 0.01

jt.printout = 1  # print the jitter process
jt.run(10)  # run with 10 random simulations
