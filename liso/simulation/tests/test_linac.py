import unittest
from unittest.mock import patch
import os.path as osp

from liso.simulation import Linac
from liso.simulation.beamline import AstraBeamline, ImpacttBeamline

_ROOT_DIR = osp.dirname(osp.abspath(__file__))


class TestLinac(unittest.TestCase):
    def testLinacConstruction(self):
        linac = Linac()

        linac.add_beamline('astra',
                           name='gun',
                           swd=_ROOT_DIR,
                           fin='injector.in',
                           template=osp.join(_ROOT_DIR, 'injector.in.000'),
                           pout='injector.0450.001')

        linac.add_beamline('impactt',
                           name='chicane',
                           swd=_ROOT_DIR,
                           fin='ImpactT.in',
                           template=osp.join(_ROOT_DIR, 'ImpactT.in.000'),
                           pout='fort.106',
                           charge=1e-15)

        beamlines = linac._beamlines
        self.assertEqual(2, len(beamlines))
        self.assertIsInstance(beamlines['gun'], AstraBeamline)
        self.assertIsInstance(beamlines['chicane'], ImpacttBeamline)

        mapping = {
            'gun_gradient': 1.,
            'MQZM1_G': 1.,
            'MQZM2_G': 1.,
        }
        # test when not all patterns are found in mapping
        with self.assertRaisesRegex(KeyError, "No mapping for"):
            linac.run(mapping)

        # TODO: test async_run

        mapping.update({
            'gun_phase': 1.,
            'charge': 0.1,
        })
        # test when keys in mapping are not found in the templates
        with self.assertRaisesRegex(ValueError, "not found in the templates"):
            linac.run(mapping)

        del mapping['charge']
        with patch.object(beamlines['gun'], 'run'):
            with patch.object(beamlines['chicane'], 'run'):
                linac.run(mapping)
