import unittest
import tempfile
import os.path as osp
import sys
from datetime import datetime

import pandas as pd
import numpy as np
import h5py

from liso.io import SimWriter
from liso.proc import Phasespace


class TestWriter(unittest.TestCase):
    def setUp(self):
        self._ps = Phasespace(
            pd.DataFrame(np.ones((100, 7)),
                         columns=['x', 'px', 'y', 'py', 'z', 'pz', 't']), 1.0)

    def testWrite(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = osp.join(tmp_dir, "tmp.hdf5")

            writer = SimWriter(10, 100, filename)
            fp_h5 = writer._fp

            # test failure when hdf5 file is already initialized
            with self.assertRaises(OSError):
                SimWriter(10, 100, filename)

            # 'datetime.fromisoformat' was introduced since Python 3.7
            if sys.version_info.minor >= 7:
                # test metadata created at initialization
                create_date = datetime.fromisoformat(
                    fp_h5['METADATA/createDate'][()])
                self.assertLess((create_date - datetime.now()).total_seconds(), 1)
                update_date = datetime.fromisoformat(
                    fp_h5['METADATA/updateDate'][()])
                self.assertLess((update_date - datetime.now()).total_seconds(), 1)

            with self.assertRaisesRegex(ValueError, 'out of range'):
                writer.write(10,
                             {'gun/gun_gradient': 1, 'gun/gun_phase': 2},
                             {'out': self._ps})

            writer.write(9,
                         {'gun/gun_gradient': 1, 'gun/gun_phase': 2},
                         {'out': self._ps})

            if sys.version_info.minor >= 7:
                # test updateDate get updated
                update_date_new = datetime.fromisoformat(
                    writer._fp['METADATA/updateDate'][()])
                self.assertLess((update_date_new - datetime.now()).total_seconds(), 1)
                self.assertNotEqual(update_date_new, update_date)

            self.assertSetEqual({'gun/gun_gradient', 'gun/gun_phase'},
                                set(fp_h5['METADATA/controlChannel']))
            np.testing.assert_array_equal(
                [0] * 9 + [10], fp_h5['INDEX/simId'][()])
            np.testing.assert_array_equal(
                [0] * 9 + [1], fp_h5['CONTROL/gun/gun_gradient'][()])
            np.testing.assert_array_equal(
                [0] * 9 + [2], fp_h5['CONTROL/gun/gun_phase'][()])
            for col in ['x', 'y', 'z', 'px', 'py', 'pz', 't']:
                np.testing.assert_array_equal(
                    self._ps[col], fp_h5[f'PHASESPACE/{col.upper()}/out'][9, ...])

            # ------------------------------
            # test writer with particle loss
            # ------------------------------

            filename2 = osp.join(tmp_dir, "tmp2.hdf5")
            with SimWriter(10, 101, filename2) as writer2:
                writer2.write(0,
                             {'gun/gun_gradient': 1, 'gun/gun_phase': 2},
                             {'out': self._ps})

                for col in ['x', 'y', 'z', 'px', 'py', 'pz', 't']:
                    self.assertFalse(
                        np.any(writer2._fp[f'PHASESPACE/{col.upper()}/out'][0, ...]))
