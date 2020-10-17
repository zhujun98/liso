import unittest
from unittest.mock import patch
import os.path as osp
import tempfile
import asyncio

import pandas as pd
import numpy as np
import h5py

from liso.io import open_sim
from liso.scan import LinacScan
from liso.simulation import Linac
from liso.data_processing import Phasespace

_ROOT_DIR = osp.dirname(osp.abspath(__file__))
_INPUT_DIR = osp.join(_ROOT_DIR, "../../simulation/tests")


class TestLinacScan(unittest.TestCase):
    def run(self, result=None):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._tmp_dir = tmp_dir
            linac = Linac()
            linac.add_beamline(
                'astra',
                name='gun',
                swd=_ROOT_DIR,
                fin='injector.in',
                template=osp.join(_INPUT_DIR, 'injector.in.000'),
                pout='injector.0450.001')
            self._sc = LinacScan(linac)
            super().run(result)

    def testParams(self):
        self._sc.add_param("param1", -0.1, 0.1, 2)
        self._sc.add_param("param2", -1., 1., 3)
        self._sc.add_param("param3",  3.,  4., 2)
        self._sc.add_param("param4", -1., sigma=0.1)  # jitter parameter
        self._sc.add_param("param5", 0., 1.)  # sample parameter
        with self.assertRaises(ValueError):
            self._sc.add_param("param4")
        self._sc.summarize()

        lst = self._sc._generate_param_sequence(2, seed=42)
        self.assertListEqual([
            (-0.1, -1.0, 3.0, -0.9503285846988767, 0.5924145688620425),
            (-0.1, -1.0, 4.0, -1.0138264301171185, 0.046450412719997725),
            (-0.1,  0.0, 3.0, -0.9352311461899308, 0.6075448519014384),
            (-0.1,  0.0, 4.0, -0.8476970143591974, 0.17052412368729153),
            (-0.1,  1.0, 3.0, -1.0234153374723336, 0.06505159298527952),
            (-0.1,  1.0, 4.0, -1.023413695694918,  0.9488855372533332),
            ( 0.1, -1.0, 3.0, -0.8420787184492609, 0.9656320330745594),
            ( 0.1, -1.0, 4.0, -0.9232565270847091, 0.8083973481164611),
            ( 0.1,  0.0, 3.0, -1.0469474385934951, 0.3046137691733707),
            ( 0.1,  0.0, 4.0, -0.9457439956414035, 0.09767211400638387),
            ( 0.1,  1.0, 3.0, -1.0463417692812462, 0.6842330265121569),
            ( 0.1,  1.0, 4.0, -1.0465729753570256, 0.4401524937396013)] + [
            (-0.1, -1.0, 3.0, -0.9758037728433966, 0.12203823484477883),
            (-0.1, -1.0, 4.0, -1.1913280244657798, 0.4951769101112702),
            (-0.1,  0.0, 3.0, -1.1724917832513033, 0.034388521115218396),
            (-0.1,  0.0, 4.0, -1.0562287529240972, 0.9093204020787821),
            (-0.1,  1.0, 3.0, -1.1012831120334423, 0.2587799816000169),
            (-0.1,  1.0, 4.0, -0.9685752667404726, 0.662522284353982),
            ( 0.1, -1.0, 3.0, -1.090802407552121, 0.31171107608941095),
            ( 0.1, -1.0, 4.0, -1.1412303701335291, 0.5200680211778108),
            ( 0.1,  0.0, 3.0, -0.8534351231078445, 0.5467102793432796),
            ( 0.1,  0.0, 4.0, -1.0225776300486535, 0.18485445552552704),
            ( 0.1,  1.0, 3.0, -0.9932471795312077, 0.9695846277645586),
            ( 0.1,  1.0, 4.0, -1.1424748186213456, 0.7751328233611146),
        ], lst)

    def testJitterParams(self):
        n = 1000
        self._sc.add_param("param1", -0.1, sigma=0.01)
        self._sc.add_param("param2", -10., sigma=-0.1)
        lst = self._sc._generate_param_sequence(n, seed=None)
        self.assertEqual(n, len(lst))
        self.assertEqual(2, len(lst[0]))

        lst1, lst2 = zip(*lst)
        self.assertLess(abs(0.01 - np.std(lst1)), 0.001)
        self.assertLess(abs(1 - np.std(lst2)), 0.1)

    def testSampleParm(self):
        n = 10
        self._sc.add_param("param1", -0.1, 0.1)
        self._sc.add_param("param2", -10., 20)
        lst = self._sc._generate_param_sequence(n, seed=None)
        self.assertEqual(n, len(lst))
        self.assertEqual(2, len(lst[0]))

        lst1, lst2 = zip(*lst)
        self.assertTrue(np.all(np.array(lst1) >= -0.1))
        self.assertTrue(np.all(np.array(lst1) < 0.1))
        self.assertTrue(np.all(np.array(lst2) >= -10))
        self.assertTrue(np.all(np.array(lst2) < 20))

    def testScan(self):
        self._sc.add_param('gun_gradient', 1., 3., num=3)
        self._sc.add_param('gun_phase', 10., 30., num=3)

        with patch.object(self._sc._linac['gun'], 'async_run') as patched_run:
            future = asyncio.Future()
            future.set_result(Phasespace(pd.DataFrame(
                columns=['x', 'px', 'y', 'py', 'z', 'pz', 't']), 0.1))
            patched_run.return_value = future
            with tempfile.NamedTemporaryFile(suffix=".hdf5") as fp:
                # Note: use n_tasks > 1 here to track bugs
                self._sc.scan(n_tasks=2, cycles=2, output=fp.name, n_particles=0)
                # Testing with a real file is necessary to check the
                # expected results were written.
                with h5py.File(fp.name, 'r') as fp_h5:
                    sim = open_sim(fp.name)
                    self.assertSetEqual(
                        {'gun/gun_gradient', 'gun/gun_phase'}, sim.control_channels)
                    self.assertSetEqual(
                        {'gun/out'}, sim.phasespace_channels)
                    np.testing.assert_array_equal(np.arange(1, 19), sim.sim_ids)
                    np.testing.assert_array_equal(
                        [1., 1., 1., 2., 2., 2., 3., 3., 3.] * 2,
                        sim.get_controls()['gun/gun_gradient'])
                    np.testing.assert_array_equal(
                        [10., 20., 30., 10., 20., 30., 10., 20., 30.] * 2,
                        sim.get_controls()['gun/gun_phase'])

            with tempfile.NamedTemporaryFile(suffix=".hdf5") as fp:
                with self.assertRaises(ValueError):
                    self._sc.scan(n_tasks=2, cycles=2, output=fp.name,
                                  n_particles=0, start_id=0)

            with tempfile.NamedTemporaryFile(suffix=".hdf5") as fp:
                # test the default value: n_tasks == None
                self._sc.scan(cycles=2, output=fp.name, n_particles=0, start_id=11)
                sim = open_sim(fp.name)
                np.testing.assert_array_equal(np.arange(1, 19) + 10, sim.sim_ids)
