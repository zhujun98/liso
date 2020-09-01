import unittest
from unittest.mock import patch
import os.path as osp
import asyncio
import tempfile

from liso.io import TempSimulationDirectory
from liso.simulation import Linac
from liso.simulation.beamline import AstraBeamline, ImpacttBeamline
from liso.simulation.output import OutputData

_ROOT_DIR = osp.dirname(osp.abspath(__file__))


class TestBeamline(unittest.TestCase):
    def setUp(self):
        linac = Linac()

        linac.add_beamline('astra',
                           name='gun',
                           swd=_ROOT_DIR,
                           fin='injector.in',
                           template=osp.join(_ROOT_DIR, 'injector.in.000'),
                           pout='injector.0450.001')

        self._bl = next(iter(linac._beamlines.values()))

    def testUpdateOutput(self):
        with self.assertRaisesRegex(RuntimeError, "Output file"):
            self._bl._update_output()

        with patch.object(self._bl, '_check_file'):
            with patch.object(self._bl, '_parse_phasespace') as patched:
                self._bl._update_output()

                patched.assert_called_once_with(
                    osp.join(_ROOT_DIR, "injector.0450.001"))
                patched.reset_mock()

                self._bl._update_output("tmp")
                patched.assert_called_once_with("tmp/injector.0450.001")
                patched.reset_mock()


class TestLinacOneBeamLine(unittest.TestCase):

    def run(self, result=None):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._tmp_dir = tmp_dir
            self._mapping = {
                'gun_gradient': 1.,
                'gun_phase': 2.,
            }
            self._linac = Linac()
            self._linac.add_beamline(
                'astra',
                name='gun',
                swd=tmp_dir,
                fin='injector.in',
                template=osp.join(_ROOT_DIR, 'injector.in.000'),
                pout='injector.0450.001')
            super().run(result)

    def testCompile(self):

        self.assertDictEqual({
            'gun.gun_gradient': 1.0, 'gun.gun_phase': 2.0
        }, self._linac.compile(self._mapping))

    @patch('liso.simulation.beamline.Beamline._run_core')
    def testRun(self, mocked_run_core):
        with patch('liso.simulation.beamline.Beamline.reset') as mocked_reset:
            with patch('liso.simulation.beamline.Beamline._update_output') as mocked_uo:
                with patch('liso.simulation.beamline.Beamline._update_statistics') as mocked_us:
                    self._linac.run(self._mapping, n_workers=1, timeout=60)
                    mocked_reset.assert_called_once_with()
                    mocked_uo.assert_called_once_with()
                    mocked_us.assert_called_once_with()

    @patch('liso.simulation.beamline.Beamline._async_run_core')
    def testAsyncRun(self, mocked_async_run_core):
        loop = asyncio.get_event_loop()
        with patch('liso.simulation.beamline.Beamline.reset') as mocked_reset:
            with patch('liso.simulation.beamline.Beamline._update_output') as mocked_uo:
                future = asyncio.Future()
                future.set_result(OutputData(dict(), dict()))
                mocked_async_run_core.return_value = future

                loop.run_until_complete(self._linac.async_run(0, self._mapping, "tmp0001"))
                tmp_dir = osp.join(self._tmp_dir, 'tmp0001')
                mocked_reset.assert_called_once_with(tmp_dir)
                mocked_uo.assert_called_once_with(tmp_dir)


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
        self.assertDictEqual({
            'gun.gun_gradient': 1.0, 'gun.gun_phase': 1.0,
            'chicane.MQZM1_G': 1.0, 'chicane.MQZM2_G': 1.0,
        }, self._linac.compile(mapping))

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
