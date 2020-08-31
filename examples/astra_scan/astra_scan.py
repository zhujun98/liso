"""
This is a basic example showing how to study parameter_scan.
"""
import numpy as np

from liso import Linac, LinacScan
from liso.logging import logger
logger.setLevel('DEBUG')


linac = Linac()  # instantiate a Linac

linac.add_beamline('astra',
                   name='gun',
                   swd='../astra_files',
                   fin='injector.in',
                   template='injector.in.000',
                   pout='injector.0450.001')

sc = LinacScan(linac)

sc.add_param('gun_gradient', 120, 140, num=2)
sc.add_param('gun_phase', -10, 10, num=2)
sc.add_param('tws_gradient', 25, 35, num=3)
sc.add_param('tws_phase', -90, -60, num=3)

sc.scan(n_tasks=4)
