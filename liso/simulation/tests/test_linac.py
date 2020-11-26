import unittest
from unittest.mock import patch
import os
import os.path as osp
import asyncio
import tempfile

from liso import Linac
from liso.config import config
from liso.io import TempSimulationDirectory
from liso.exceptions import LisoRuntimeError
from liso.simulation.beamline import AstraBeamline, ImpacttBeamline

_ROOT_DIR = osp.dirname(osp.abspath(__file__))


class TestAstraBeamline(unittest.TestCase):
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
        with self.assertRaisesRegex(LisoRuntimeError, "Output file"):
            self._bl._update_output(_ROOT_DIR)

        with patch.object(self._bl, '_check_file'):
            with patch.object(self._bl, '_parse_phasespace') as patched:
                self._bl._update_output(_ROOT_DIR)

                patched.assert_called_once_with(
                    osp.join(_ROOT_DIR, "injector.0450.001"))
                patched.reset_mock()

                self._bl._update_output("tmp")
                patched.assert_called_once_with("tmp/injector.0450.001")
                patched.reset_mock()

    @patch.dict(config['EXECUTABLE'], {"ASTRA": "astra_fake"})
    def testCheckExecutable(self):
        with self.assertRaisesRegex(AssertionError, "executable .astra_fake. is not available"):
            self._bl._check_executable()


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
            'gun/gun_gradient': 1.0, 'gun/gun_phase': 2.0
        }, self._linac.compile(self._mapping))

    @patch('liso.simulation.beamline.Beamline._run_core')
    def testRun(self, mocked_run_core):
        with patch('liso.simulation.beamline.Beamline.reset') as mocked_reset:
            with patch('liso.simulation.beamline.Beamline._update_output') as mocked_uo:
                with patch('liso.simulation.beamline.Beamline._update_statistics') as mocked_us:
                    with patch('liso.simulation.beamline.Beamline._generate_initial_particle_file') as mocked_gipf:
                        self._linac.run(self._mapping, n_workers=1, timeout=60)
                        mocked_reset.assert_called_once_with()
                        mocked_uo.assert_called_once_with(self._tmp_dir)
                        mocked_us.assert_called_once_with()
                        mocked_gipf.assert_not_called()

    @patch('liso.simulation.beamline.Beamline._async_run_core')
    def testAsyncRun(self, mocked_async_run_core):
        loop = asyncio.get_event_loop()
        with patch('liso.simulation.beamline.Beamline._update_output') as mocked_uo:
            with patch('liso.simulation.beamline.Beamline._generate_initial_particle_file') as mocked_gipf:

                future = asyncio.Future()
                future.set_result(object())
                mocked_async_run_core.return_value = future

                tmp_dir = osp.join(self._tmp_dir, "tmp000001")
                # os.mkdir(tmp_dir)
                sim_id, controls, phasespaces = loop.run_until_complete(
                    self._linac.async_run(1, self._mapping))

                mocked_uo.assert_called_once_with(tmp_dir)
                mocked_gipf.assert_not_called()

                self.assertEqual(1, sim_id)
                self.assertDictEqual({'gun/gun_gradient': 1.0, 'gun/gun_phase': 2.0}, controls)
                self.assertDictEqual({'gun/out': mocked_uo()}, phasespaces)


class TestLinacTwoBeamLine(unittest.TestCase):
    def run(self, result=None):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._tmp_dir = tmp_dir
            self._mapping = {
                'gun/gun_gradient': 1.,
                'gun/gun_phase': 1.,
                'chicane/MQZM1_G': 1.,
                'chicane/MQZM2_G': 1.,
            }
            self._linac = Linac()

            self._linac.add_beamline('astra',
                                     name='gun',
                                     swd=tmp_dir,
                                     fin='injector.in',
                                     template=osp.join(_ROOT_DIR, 'injector.in.000'),
                                     pout='injector.0450.001')

            self._linac.add_beamline('impactt',
                                     name='chicane',
                                     swd=tmp_dir,
                                     fin='ImpactT.in',
                                     template=osp.join(_ROOT_DIR, 'ImpactT.in.000'),
                                     pout='fort.106',
                                     charge=1e-15)

            beamlines = self._linac._beamlines
            self.assertEqual(2, len(beamlines))
            self.assertIsInstance(beamlines['gun'], AstraBeamline)
            self.assertIsInstance(beamlines['chicane'], ImpacttBeamline)

            super().run(result)

    def testCompile(self):
        mapping = self._mapping.copy()

        # test when not all patterns are found in mapping
        del mapping['gun/gun_phase']
        with self.assertRaisesRegex(KeyError, "No mapping for <gun_phase>"):
            self._linac.compile(mapping)

        # test when keys in mapping are not found in the templates
        mapping['gun/gun_phase'] = 1.0
        mapping['gun/charge'] = 1.0
        with self.assertRaisesRegex(KeyError, "{'charge'} not found in the templates"):
            self._linac.compile(mapping)

        del mapping['gun/charge']
        self.assertDictEqual({
            'gun/gun_gradient': 1.0, 'gun/gun_phase': 1.0,
            'chicane/MQZM1_G': 1.0, 'chicane/MQZM2_G': 1.0,
        }, self._linac.compile(mapping))

    @patch('liso.simulation.beamline.Beamline._update_statistics')
    @patch('liso.simulation.beamline.Beamline._run_core')
    def testRun(self, mocked_run_core, mocked_update_statistics):
        mapping = self._mapping
        with patch.object(
                self._linac['gun'], '_update_output') as mocked_gun_uo:
            with patch.object(
                    self._linac['chicane'], '_update_output') as mocked_chicane_uo:
                with patch.object(
                        self._linac['gun'], '_generate_initial_particle_file') as mocked_gun_gipf:
                    with patch.object(
                            self._linac['chicane'], '_generate_initial_particle_file') as mocked_chicane_gipf:
                        self._linac.run(mapping)

                        self.assertEqual(2, mocked_run_core.call_count)
                        self.assertEqual(2, mocked_update_statistics.call_count)
                        mocked_gun_uo.assert_called_once_with(self._tmp_dir)
                        mocked_chicane_uo.assert_called_once_with(self._tmp_dir)
                        mocked_gun_gipf.assert_not_called()
                        mocked_chicane_gipf.assert_called_once_with(mocked_gun_uo(), self._tmp_dir)

    @patch('liso.simulation.beamline.Beamline._async_run_core')
    def testAsyncRun(self, mocked_async_run_core):
        sim_id_gt = 10
        mapping = self._mapping

        loop = asyncio.get_event_loop()
        with patch.object(
                self._linac['gun'], '_update_output') as mocked_gun_uo:
            with patch.object(
                    self._linac['chicane'], '_update_output') as mocked_chicane_uo:
                with patch.object(
                        self._linac['gun'],
                        '_generate_initial_particle_file') as mocked_gun_gipf:
                    with patch.object(
                            self._linac['chicane'],
                            '_generate_initial_particle_file') as mocked_chicane_gipf:

                        future = asyncio.Future()
                        future.set_result(object())
                        mocked_async_run_core.return_value = future

                        sim_id, controls, phasespaces = loop.run_until_complete(
                            self._linac.async_run(sim_id_gt, mapping))

                        self.assertEqual(2, mocked_async_run_core.call_count)
                        actual_tmp_dir = osp.join(self._tmp_dir, f"tmp0000{sim_id_gt}")
                        mocked_gun_uo.assert_called_once_with(actual_tmp_dir)
                        mocked_chicane_uo.assert_called_once_with(actual_tmp_dir)

                        mocked_gun_gipf.assert_not_called()
                        mocked_chicane_gipf.assert_called_once_with(
                            mocked_gun_uo(), actual_tmp_dir)

                        self.assertEqual(sim_id_gt, sim_id)
                        self.assertDictEqual(
                            {'gun/gun_gradient': 1.0, 'gun/gun_phase': 1.0,
                             'chicane/MQZM1_G': 1.0, 'chicane/MQZM2_G': 1.0}, controls)
                        self.assertDictEqual(
                            {'gun/out': mocked_gun_uo(),
                             'chicane/out': mocked_chicane_uo()}, phasespaces)
