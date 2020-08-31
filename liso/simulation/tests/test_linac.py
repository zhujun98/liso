import unittest
from unittest.mock import patch
import os.path as osp
import asyncio

from liso.io import TempSimulationDirectory
from liso.simulation import Linac
from liso.simulation.beamline import AstraBeamline, ImpacttBeamline
from liso.simulation.output import OutputData

_ROOT_DIR = osp.dirname(osp.abspath(__file__))


class TestLinacOneBeamLine(unittest.TestCase):
    def setUp(self):
        self._linac = Linac()

        self._linac.add_beamline('astra',
                                 name='gun',
                                 swd=_ROOT_DIR,
                                 fin='injector.in',
                                 template=osp.join(_ROOT_DIR, 'injector.in.000'),
                                 pout='injector.0450.001')

    def testCompile(self):
        mapping = {
            'gun_gradient': 1.,
            'gun_phase': 2.,
        }
        self._linac.compile(mapping)


class TestLinacTwoBeamLine(unittest.TestCase):
    def setUp(self):
        self._linac = Linac()

        self._linac.add_beamline('astra',
                                 name='gun',
                                 swd=_ROOT_DIR,
                                 fin='injector.in',
                                 template=osp.join(_ROOT_DIR, 'injector.in.000'),
                                 pout='injector.0450.001')

        self._linac.add_beamline('impactt',
                                 name='chicane',
                                 swd=_ROOT_DIR,
                                 fin='ImpactT.in',
                                 template=osp.join(_ROOT_DIR, 'ImpactT.in.000'),
                                 pout='fort.106',
                                 charge=1e-15)

        beamlines = self._linac._beamlines
        self.assertEqual(2, len(beamlines))
        self.assertIsInstance(beamlines['gun'], AstraBeamline)
        self.assertIsInstance(beamlines['chicane'], ImpacttBeamline)

    def testCompile(self):
        mapping = {
            'gun.gun_gradient': 1.,
            'chicane.MQZM1_G': 1.,
            'chicane.MQZM2_G': 1.,
        }
        # test when not all patterns are found in mapping
        with self.assertRaisesRegex(KeyError, "No mapping for"):
            self._linac.compile(mapping)

        mapping.update({
            'gun.gun_phase': 1.,
            'gun.charge': 0.1,
        })
        # test when keys in mapping are not found in the templates
        with self.assertRaisesRegex(KeyError, "not found in the templates"):
            self._linac.compile(mapping)

        del mapping['gun.charge']
        self._linac.compile(mapping)

    def testRun(self):
        mapping = dict()
        with patch.object(self._linac, 'compile') as patched_compile:
            with patch.object(self._linac['gun'], 'run') as patched_gun_run:
                with patch.object(self._linac['chicane'], 'run') as patched_chicane_run:
                    self._linac.run(mapping)
                    patched_compile.assert_called_once_with(mapping)
                    patched_gun_run.assert_called_once_with(1, None)
                    patched_chicane_run.assert_called_once_with(1, None)

    def testSyncRun(self):
        index = 10
        mapping = dict()

        loop = asyncio.get_event_loop()
        with TempSimulationDirectory('temp_dir') as tmp_dir:
            with patch.object(self._linac, 'compile') as patched_compile:
                with patch.object(self._linac['gun'], 'async_run') \
                        as patched_gun_run:
                    with patch.object(self._linac['chicane'], 'async_run') \
                            as patched_chicane_run:
                        future1 = asyncio.Future()
                        future1.set_result(OutputData(dict(), dict()))
                        patched_gun_run.return_value = future1

                        future2 = asyncio.Future()
                        future2.set_result(OutputData(dict(), dict()))
                        patched_chicane_run.return_value = future2

                        loop.run_until_complete(
                            self._linac.async_run(index, mapping, tmp_dir))
                        patched_compile.assert_called_once_with(mapping)
                        patched_gun_run.assert_called_once_with(
                            tmp_dir, timeout=None)
                        patched_chicane_run.assert_called_once_with(
                            tmp_dir, timeout=None)
