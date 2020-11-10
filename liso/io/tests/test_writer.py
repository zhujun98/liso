import unittest
import tempfile
import os.path as osp
from datetime import datetime

import h5py
import pandas as pd
import numpy as np

from liso import EuXFELInterface
from liso import doocs_channels as dc
from liso.io import ExpWriter, SimWriter
from liso.proc import Phasespace


def _check_create_update_date(fp):
    create_date = datetime.fromisoformat(fp['METADATA/createDate'][()])
    assert 0 < (datetime.now() - create_date).total_seconds() < 0.1
    update_date = datetime.fromisoformat(fp['METADATA/updateDate'][()])
    assert 0 < (datetime.now() - update_date).total_seconds() < 0.1
    assert (update_date - create_date).total_seconds() > 0


class TestSimWriter(unittest.TestCase):
    def setUp(self):
        self._ps = Phasespace(
            pd.DataFrame(np.ones((100, 7)),
                         columns=['x', 'px', 'y', 'py', 'z', 'pz', 't']), 1.0)

    def testWrite(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = osp.join(tmp_dir, "tmp.hdf5")

            with SimWriter(10, 100, filename, start_id=100,
                           chunk_size=5, max_size_per_file=100) as writer:

                for i, idx in enumerate(range(9)):
                    writer.write(idx,
                                 {'gun/gun_gradient': 10 * i, 'gun/gun_phase': 0.1 * i},
                                 {'out': self._ps})

                with self.assertRaisesRegex(ValueError, 'out of range'):
                    writer.write(10,
                                 {'gun/gun_gradient': 1, 'gun/gun_phase': 2},
                                 {'out': self._ps})

            with self.subTest("Initialize writer when hdf5 file already exits"):
                with self.assertRaises(OSError):
                    SimWriter(10, 100, filename)

            with self.subTest("Test data in the file"):
                with h5py.File(filename, 'r') as fp:
                    _check_create_update_date(fp)

                    self.assertSetEqual({'gun/gun_gradient', 'gun/gun_phase'},
                                        set(fp['METADATA/controlChannel']))
                    self.assertSetEqual({'out'},
                                        set(fp['METADATA/phasespaceChannel']))

                    np.testing.assert_array_equal(
                        [100, 101, 102, 103, 104, 105, 106, 107, 108,   0],
                        fp['INDEX/simId'][()])

                    np.testing.assert_array_equal(
                        [ 0., 10., 20., 30., 40., 50., 60., 70., 80.,  0.],
                        fp['CONTROL/gun/gun_gradient'][()])
                    np.testing.assert_array_almost_equal(
                        [0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0. ],
                        fp['CONTROL/gun/gun_phase'][()])

                    for i in range(10):
                        for col in ['x', 'y', 'z', 'px', 'py', 'pz', 't']:
                            if i == 9:
                                self.assertTrue(
                                    np.all(fp[f'PHASESPACE/{col.upper()}/out'][i, ...] == 0))
                            else:
                                np.testing.assert_array_equal(
                                    self._ps[col], fp[f'PHASESPACE/{col.upper()}/out'][i, ...])

            with self.subTest("Writer with particle loss"):
                filename2 = osp.join(tmp_dir, "tmp2.hdf5")
                with SimWriter(10, 101, filename2) as writer2:
                    writer2.write(0,
                                 {'gun/gun_gradient': 1, 'gun/gun_phase': 2},
                                 {'out': self._ps})

                    for col in ['x', 'y', 'z', 'px', 'py', 'pz', 't']:
                        self.assertFalse(
                            np.any(writer2._fp[f'PHASESPACE/{col.upper()}/out'][0, ...]))


class TestExpWriter(unittest.TestCase):
    def setUp(self):
        m = EuXFELInterface()
        m.add_control_channel(dc.FLOAT64, "A/B/C/D")
        m.add_control_channel(dc.FLOAT32, "A/B/C/E")
        m.add_instrument_channel(dc.IMAGE, "H/I/J/K", shape=(4, 4), dtype='uint16')
        m.add_instrument_channel(dc.IMAGE, "H/I/J/L", shape=(5, 6), dtype='float32')
        self._schema = m.schema

    def testWrite(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = osp.join(tmp_dir, "tmp.hdf5")
            chunk_size = 10
            s1 = (4, 4)
            s2 = (5, 6)

            with ExpWriter(filename, schema=self._schema,
                           chunk_size=chunk_size) as writer:
                # test failure when hdf5 file is already initialized
                with self.assertRaises(OSError):
                    ExpWriter(filename, schema=self._schema)

                for i, pid in enumerate(np.arange(9) + 100):
                    writer.write(
                        pid,
                        {"A/B/C/D": 10 * i, "A/B/C/E": 0.1 * i},
                        {"H/I/J/K": np.ones(s1, dtype=np.uint16),
                         "H/I/J/L": np.ones(s2, dtype=np.float32)}
                    )

            with self.subTest("Test data in the file"):
                with h5py.File(filename, 'r') as fp:
                    _check_create_update_date(fp)

                    self.assertSetEqual({"A/B/C/D", "A/B/C/E"},
                                        set(fp['METADATA/controlChannel']))
                    self.assertSetEqual({"H/I/J/K", "H/I/J/L"},
                                        set(fp['METADATA/instrumentChannel']))

                    np.testing.assert_array_equal(
                        [100, 101, 102, 103, 104, 105, 106, 107, 108],
                        fp['INDEX/pulseId'][()])

                    self.assertEqual(np.float64, fp['CONTROL/A/B/C/D'].dtype)
                    self.assertEqual(np.float32, fp['CONTROL/A/B/C/E'].dtype)
                    np.testing.assert_array_equal(
                        [ 0., 10., 20., 30., 40., 50., 60., 70., 80.,  0.],
                        fp['CONTROL/A/B/C/D'][()])
                    np.testing.assert_array_almost_equal(
                        [0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.],
                        fp['CONTROL/A/B/C/E'][()])

                    self.assertEqual((chunk_size, *s1), fp['INSTRUMENT/H/I/J/K'].shape)
                    self.assertEqual((chunk_size, *s2), fp['INSTRUMENT/H/I/J/L'].shape)
                    self.assertEqual(np.uint16, fp['INSTRUMENT/H/I/J/K'].dtype)
                    self.assertEqual(np.float32, fp['INSTRUMENT/H/I/J/L'].dtype)
                    np.testing.assert_array_equal(
                        np.concatenate((np.ones((chunk_size - 1, *s1)), np.zeros((1, *s1)))),
                        fp['INSTRUMENT/H/I/J/K'][()])
                    np.testing.assert_array_equal(
                        np.concatenate((np.ones((chunk_size - 1, *s2)), np.zeros((1, *s2)))),
                        fp['INSTRUMENT/H/I/J/L'][()])
