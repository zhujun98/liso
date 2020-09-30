import unittest
import tempfile

import numpy as np
import pandas as pd

from liso import Phasespace
from liso.io import SimWriter, open_sim
from liso.io.reader import DataCollection


class TestReader(unittest.TestCase):
    def testOpenSim(self):
        n_sims = 10
        ps = Phasespace(
            pd.DataFrame(np.ones((100, 7)),
                         columns=['x', 'px', 'y', 'py', 'z', 'pz', 't']), 1.0)

        with tempfile.NamedTemporaryFile(suffix=".hdf5") as fp:
            writer = SimWriter(n_sims, 100, fp.name)

            for i in range(n_sims):
                writer.write(i,
                             {'gun.gun_gradient': 10 * i, 'gun.gun_phase': 20 * i},
                             {'out': ps})

            data = open_sim(fp.name)
            self.assertIsInstance(data, DataCollection)
            self.assertSetEqual({'gun.gun_gradient', 'gun.gun_phase'},
                                data.control_sources)
            self.assertSetEqual({'out'}, data.phasespace_sources)
            self.assertListEqual([i+1 for i in range(n_sims)], data.sim_ids)

            control_data = data.get_controls()
            self.assertEqual(n_sims, len(control_data))
            self.assertEqual(['gun.gun_gradient', 'gun.gun_phase'],
                             control_data.columns.tolist())
            np.testing.assert_array_equal(10 * np.arange(10),
                                          control_data['gun.gun_gradient'])
            np.testing.assert_array_equal(20 * np.arange(10),
                                          control_data['gun.gun_phase'])
            self.assertListEqual([i+1 for i in range(n_sims)],
                                 control_data.index.tolist())
