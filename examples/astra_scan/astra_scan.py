"""
This is a basic example showing how to study parameter_scan.
"""
from liso import Linac, LinacScan
from liso.logging import logger
logger.setLevel('DEBUG')


linac = Linac(2000)

linac.add_beamline('astra',
                   name='gun',
                   swd='../astra_files',
                   fin='injector.in',
                   template='injector.in.000',
                   pout='injector.0450.001')

sc = LinacScan(linac)

sc.add_param('gun_gradient', start=120, stop=140, num=4, sigma=-0.001)
sc.add_param('gun_phase', start=-10, stop=10, num=3, sigma=0.1)
sc.add_param('tws_gradient', lb=25, ub=35)
sc.add_param('tws_phase', value=-90, sigma=0.1)

sc.scan()
