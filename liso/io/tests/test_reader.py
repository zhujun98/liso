import unittest
import tempfile

import numpy as np
import pandas as pd

from liso import Phasespace
from liso.io import SimWriter, open_run, open_sim
from liso.io.reader import ExpDataCollection, SimDataCollection


class TestReader(unittest.TestCase):
    def testOpenSim(self):
        n_sims = 10

        with tempfile.NamedTemporaryFile(suffix=".hdf5") as fp:
            writer = SimWriter(n_sims, 100, fp.name)

            for i in range(n_sims):
                ps1 = Phasespace(
                    pd.DataFrame(np.ones((100, 7)) * i,
                                 columns=['x', 'px', 'y', 'py', 'z', 'pz', 't']), 1.0)

                ps2 = Phasespace(
                    pd.DataFrame(np.ones((100, 7)) * i * 10,
                                 columns=['x', 'px', 'y', 'py', 'z', 'pz', 't']), 1.0)
                writer.write(i,
                             {'gun.gun_gradient': 10 * i, 'gun.gun_phase': 20 * i},
                             {'out1': ps1, 'out2': ps2})

            data = open_sim(fp.name)
            data.info()
            self.assertIsInstance(data, SimDataCollection)
            self.assertSetEqual({'gun.gun_gradient', 'gun.gun_phase'},
                                data.control_channels)
            self.assertSetEqual({'out1', 'out2'}, data.phasespace_channels)
            self.assertListEqual([i+1 for i in range(n_sims)], data.sim_ids)

            # test "get_controls" method
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

            # test access data by simId
            with self.assertRaises(IndexError):
                data[0]

            for sid, sim in data:
                self.assertSetEqual({
                    'gun.gun_phase', 'gun.gun_gradient', 'out1', 'out2'
                }, set(sim.keys()))
                i = sid - 1
                self.assertEqual(20 * i, sim['gun.gun_phase'])
                self.assertEqual(10 * i, sim['gun.gun_gradient'])
                np.testing.assert_array_equal(np.ones(100) * i, sim['out1']['x'])
                np.testing.assert_array_equal(np.ones(100) * 10 * i, sim['out2']['y'])

            # test access data by index

            sim_id, sim = data.iloc(1)
            self.assertEqual(2, sim_id)
            self.assertSetEqual({
                'gun.gun_phase', 'gun.gun_gradient', 'out1', 'out2'
            }, set(sim.keys()))
            self.assertEqual(20 * 1, sim['gun.gun_phase'])
            self.assertEqual(10 * 1, sim['gun.gun_gradient'])
            np.testing.assert_array_equal(np.ones(100) * 1, sim['out1']['x'])
            np.testing.assert_array_equal(np.ones(100) * 10 * 1, sim['out2']['y'])

    def testOpenRun(self):
        pass
