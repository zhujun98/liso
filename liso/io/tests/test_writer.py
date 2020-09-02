import unittest
import tempfile
import pandas as pd
import numpy as np
import h5py

from liso.io import SimWriter
from liso.data_processing import Phasespace
from liso.simulation.output import OutputData


class TestWriter(unittest.TestCase):
    def testWrite(self):
        with tempfile.NamedTemporaryFile(suffix=".hdf5") as fp:
            writer = SimWriter(10, 1000, fp.name)

            with self.assertRaisesRegex(ValueError, 'out of range'):
                writer.write(10, OutputData(
                    {'gun.gun_gradient': 1, 'gun.gun_phase': 2},
                    {'out': Phasespace(pd.DataFrame(), 0.1)}
                ))

            writer.write(9, OutputData(
                {'gun.gun_gradient': 1, 'gun.gun_phase': 2},
                {'out': Phasespace(pd.DataFrame(), 0.1)}
            ))

            with h5py.File(fp.name, 'r') as fp_h5:
                self.assertSetEqual(
                    {'gun.gun_gradient', 'gun.gun_phase'},
                    set(fp_h5['metadata']['input']))
                np.testing.assert_array_equal(
                    [0] * 9 + [1], fp_h5['input']['gun.gun_gradient'][()])
                np.testing.assert_array_equal(
                    [0] * 9 + [2], fp_h5['input']['gun.gun_phase'][()])
