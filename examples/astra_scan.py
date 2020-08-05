"""
This is a basic example showing how to study parameter_scan.
"""
import numpy as np

from liso import Linac, LinacScan


linac = Linac()  # instantiate a Linac

linac.add_beamline('astra',
                   name='gun',
                   fin='astra_injector/injector.in',
                   template='astra_jitter/injector.in.000',
                   pout='injector.0450.001')

sc = LinacScan(linac)  # instantiate a Jitter (problem)

n = 5
sc.add_param('gun_gradient', values=np.linspace(130 - 0.06, 130 + 0.06, n))
sc.add_param('gun_phase', values=np.linspace(-0.02, 0.02, n))
sc.add_param('tws_gradient', values=np.linspace(30 - 0.06, 30 + 0.06, n))
sc.add_param('tws_phase', values=np.linspace(-0.02, 0.02, n))

sc.printout = 1  # print the parameter_scan process
sc.scan()
