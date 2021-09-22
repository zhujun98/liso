"""
This is a basic example showing how to perform parameter scan in ASTRA.
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cluster", action="store_true")

args = parser.parse_args()
if args.cluster:
    n1, n2 = 40, 30
else:
    n1, n2 = 4, 3


from liso import Linac, LinacScan


linac = Linac(2000)

linac.add_beamline('astra',
                   name='gun',
                   swd='../astra_files',
                   fin='injector.in',
                   template='injector.in.000',
                   pout='injector.0450.001')

sc = LinacScan(linac)

sc.add_param('gun_gradient', start=120, stop=140, num=n1, sigma=-0.001)
sc.add_param('gun_phase', start=-10, stop=10, num=n2, sigma=0.1)
sc.add_param('tws_gradient', lb=25, ub=35)
sc.add_param('tws_phase', value=-90, sigma=0.1)

sc.scan()
