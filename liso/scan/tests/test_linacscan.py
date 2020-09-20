import unittest
from unittest.mock import patch
import os.path as osp
import tempfile
import asyncio

import pandas as pd
import numpy as np
import h5py

from liso.scan import LinacScan
from liso.simulation import Linac
from liso.data_processing import Phasespace

_ROOT_DIR = osp.dirname(osp.abspath(__file__))
_INPUT_DIR = osp.join(_ROOT_DIR, "../../simulation/tests")


class TestLinacscan(unittest.TestCase):
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

    def testScanParams(self):
        self._sc.add_param("param1", -0.1, 0.1, 2)
        self._sc.add_param("param2", -1., 1., 3)
        self._sc.add_param("param3",  3.,  4., 2)
        self._sc.add_param("param4", -1.)
        with self.assertRaises(ValueError):
            self._sc.add_param("param4")

        lst = self._sc._generate_param_sequence(2)
        self.assertListEqual([
            (-0.1, -1.0, 3.0, -1.0), (-0.1, -1.0, 4.0, -1.0), (-0.1, 0.0, 3.0, -1.0),
            (-0.1,  0.0, 4.0, -1.0), (-0.1,  1.0, 3.0, -1.0), (-0.1, 1.0, 4.0, -1.0),
            ( 0.1, -1.0, 3.0, -1.0), ( 0.1, -1.0, 4.0, -1.0), ( 0.1, 0.0, 3.0, -1.0),
            ( 0.1,  0.0, 4.0, -1.0), ( 0.1,  1.0, 3.0, -1.0), ( 0.1, 1.0, 4.0, -1.0)] * 2, lst)

    def testJitterParams(self):
        n = 1000
        self._sc.add_param("param1", -0.1, sigma=0.01)
        self._sc.add_param("param2", -10., sigma=-0.1)
        lst = self._sc._generate_param_sequence(n)
        self.assertEqual(n, len(lst))
        self.assertEqual(2, len(lst[0]))

        lst1, lst2 = zip(*lst)
        self.assertLess(abs(0.01 - np.std(lst1)), 0.001)
        self.assertLess(abs(1 - np.std(lst2)), 0.1)

    def testScan(self):
        self._sc.add_param('gun_gradient', 1.)
        self._sc.add_param('gun_phase', 2.)

        with patch.object(self._sc._linac['gun'], 'async_run') as patched_run:
            future = asyncio.Future()
            future.set_result(Phasespace(pd.DataFrame(
                columns=['x', 'px', 'y', 'py', 'z', 'pz', 't']), 0.1))
            patched_run.return_value = future
            with tempfile.NamedTemporaryFile(suffix=".hdf5") as fp:
                self._sc.scan(repeat=2, output=fp.name, n_particles=0)
                # Testing with a real file is necessary to check the
                # expected results were written.
                with h5py.File(fp.name, 'r') as fp_h5:
                    self.assertSetEqual(
                        {'gun.gun_gradient', 'gun.gun_phase'},
                        set(fp_h5['metadata']['input']))
                    np.testing.assert_array_equal(
                        [1, 1], fp_h5['input']['gun.gun_gradient'][()])
                    np.testing.assert_array_equal(
                        [2, 2], fp_h5['input']['gun.gun_phase'][()])
