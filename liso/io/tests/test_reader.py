import unittest
import tempfile
import os

import numpy as np
import pandas as pd

from liso import EuXFELInterface, Phasespace
from liso.experiment import doocs_channels as dc
from liso.io import ExpWriter, SimWriter, open_run, open_sim
from liso.io.reader import ExpDataCollection, SimDataCollection


class TestSimReader(unittest.TestCase):
    def testOpenSim(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            n_sims = 10
            macroparticles = 100
            schema = (
                {
                    "gun/gun_gradient": {"type": "<f4"},
                    "gun/gun_phase": {"type": "<f4"}
                },
                {
                    "gun/out1": {
                        "macroparticles": macroparticles,
                        "type": "phasespace"
                    },
                    "gun/out2": {
                        "macroparticles": macroparticles,
                        "type": "phasespace"
                    }
                }
            )
            files = []
            chunk_size = 20

            for i_file in range(2):
                tmp_file = os.path.join(tmp_dir, f"tmp{i_file}.hdf5")

                start_id = 1 + i_file * n_sims
                with SimWriter(tmp_file,
                               start_id=start_id,
                               schema=schema,
                               chunk_size=chunk_size) as writer:
                    for j in range(n_sims):
                        ps1 = Phasespace(
                            pd.DataFrame(np.ones((100, 7)) * (j + start_id),
                                         columns=['x', 'px', 'y', 'py', 'z', 'pz', 't']), 1.0)

                        ps2 = Phasespace(
                            pd.DataFrame(np.ones((100, 7)) * (j + start_id) * 10,
                                         columns=['x', 'px', 'y', 'py', 'z', 'pz', 't']), 1.0)

                        writer.write(j,
                                     {'gun/gun_gradient': 10 * (j + start_id),
                                      'gun/gun_phase': 20 * (j + start_id)},
                                     {'gun/out1': ps1, 'gun/out2': ps2})

                files.append(tmp_file)

            with self.subTest("Test opening a single file"):
                data = open_sim(files[0])
                self.assertIsInstance(data, SimDataCollection)
                data.info()
                self._check_sim_metadata(data, 10)
                self._check_sim_get_control(data, 10)
                self._check_sim_iterate_over_data(data)
                self._check_sim_access_data_by_index(data, idx=1)

            with self.subTest("Test opening two files in a folder"):
                data = open_sim(tmp_dir)
                self._check_sim_metadata(data, 20)
                self._check_sim_get_control(data, 20)
                self._check_sim_iterate_over_data(data)
                self._check_sim_access_data_by_index(data, idx=12)

            with self.subTest("Test control channel data"):
                with self.assertRaisesRegex(KeyError, 'No data was found for channel'):
                    data.channel('gun/random')

                item = data.channel('gun/gun_phase')
                with self.assertRaises(KeyError):
                    item[0]
                with self.assertRaises(KeyError):
                    item[21]
                self.assertEqual(10 * 20, item[10])
                self.assertEqual(10 * 20, item.from_index(9))
                self.assertEqual(np.float32, item.numpy().dtype)
                np.testing.assert_array_equal(20 * np.arange(1, 21), item.numpy())

            with self.subTest("Test phasespace channel data"):
                item = data.channel('gun/out1')
                self.assertEqual(np.float64, item.numpy().dtype)
                np.testing.assert_array_equal(3 * np.ones((7, 100)), item[3])

                item = data.channel('gun/out1', 't')
                np.testing.assert_array_equal(4 * np.ones((1, 100)), item[4])

                item = data.channel('gun/out1', ['x', 'y'])
                np.testing.assert_array_equal(10 * np.ones((2, 100)), item[10])

                with self.assertRaisesRegex(ValueError, "not a valid phasespace column"):
                    data.channel('gun/out1', 'a')

                with self.assertRaisesRegex(ValueError, "not a valid phasespace column"):
                    data.channel('gun/out1', ['x', 'a'])

    def _check_sim_metadata(self, data, n):
        self.assertSetEqual({'gun/gun_gradient', 'gun/gun_phase'},
                            data.control_channels)
        self.assertSetEqual({'gun/out1', 'gun/out2'}, data.phasespace_channels)
        self.assertListEqual([i+1 for i in range(n)], data.sim_ids)

    def _check_sim_get_control(self, data, n):
        control_data = data.get_controls()
        self.assertEqual(n, len(control_data))
        self.assertEqual(['gun/gun_gradient', 'gun/gun_phase'],
                         control_data.columns.tolist())
        np.testing.assert_array_equal(10 * np.arange(1, n + 1),
                                      control_data['gun/gun_gradient'])
        np.testing.assert_array_equal(20 * np.arange(1, n + 1),
                                      control_data['gun/gun_phase'])
        self.assertListEqual([i+1 for i in range(n)], control_data.index.tolist())

    def _check_sim_iterate_over_data(self, data):
        with self.assertRaises(KeyError):
            data[0]

        with self.assertRaises(KeyError):
            data[21]

        for sid, sim in data:
            self.assertSetEqual({
                'gun/gun_phase', 'gun/gun_gradient', 'gun/out1', 'gun/out2'
            }, set(sim.keys()))
            self.assertEqual(20 * sid, sim['gun/gun_phase'])
            self.assertEqual(10 * sid, sim['gun/gun_gradient'])
            np.testing.assert_array_equal(np.ones(100) * sid, sim['gun/out1']['x'])
            np.testing.assert_array_equal(np.ones(100) * 10 * sid, sim['gun/out2']['y'])

    def _check_sim_access_data_by_index(self, data, idx):
        sim_id, sim = data.from_index(idx)
        id_ = idx + 1
        self.assertEqual(id_, sim_id)
        self.assertSetEqual({
            'gun/gun_phase', 'gun/gun_gradient', 'gun/out1', 'gun/out2'
        }, set(sim.keys()))
        self.assertEqual(20 * id_, sim['gun/gun_phase'])
        self.assertEqual(10 * id_, sim['gun/gun_gradient'])
        np.testing.assert_array_equal(np.ones(100) * id_, sim['gun/out1']['x'])
        np.testing.assert_array_equal(np.ones(100) * 10 * id_, sim['gun/out2']['y'])


class TestExpReader(unittest.TestCase):
    def testOpenRun(self):
        n_pulses = 10
        pulse_ids = np.arange(1, 2 * n_pulses, 2)
        s1 = (3, 4)
        s2 = (6, 5)

        m = EuXFELInterface()
        m.add_control_channel(dc.FLOAT32, "A/B/C/D")
        m.add_control_channel(dc.FLOAT32, "A/B/C/E")
        m.add_control_channel(dc.BOOL, "A/B/C/F")
        m.add_instrument_channel(dc.IMAGE, "H/I/J/K", shape=s1, dtype='uint16')
        m.add_instrument_channel(dc.IMAGE, "H/I/J/L", shape=s2, dtype='float32')
        self._schema = m.schema

        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = os.path.join(tmp_dir, "tmp.hdf5")
            chunk_size = 10

            with ExpWriter(filename,
                           schema=self._schema, chunk_size=chunk_size) as writer:
                for i, pid in enumerate(pulse_ids):
                    writer.write(
                        pid,
                        {"A/B/C/D": 10 * i, "A/B/C/E": 0.1 * i},
                        {"H/I/J/K": np.ones(s1, dtype=np.uint16),
                         "H/I/J/L": np.ones(s2, dtype=np.float32)}
                    )

            data = open_run(filename)
            self.assertIsInstance(data, ExpDataCollection)
            data.info()

            with self.subTest("Test control data"):
                controls = data.get_controls()
                self.assertListEqual(["A/B/C/D", "A/B/C/E", "A/B/C/F"],
                                     controls.columns.tolist())

                np.testing.assert_array_equal(pulse_ids, controls.index.to_numpy())
                np.testing.assert_array_equal(10. * np.arange(10), controls["A/B/C/D"])
                np.testing.assert_array_almost_equal(.1 * np.arange(10), controls["A/B/C/E"])
                np.testing.assert_array_equal(np.zeros(10).astype(bool), controls["A/B/C/F"])

                for i, (pid, item) in enumerate(data):
                    self.assertSetEqual({
                        "A/B/C/D", "A/B/C/E", "A/B/C/F", "H/I/J/K", "H/I/J/L"
                    }, set(item.keys()))

                    self.assertAlmostEqual(10. * i, item["A/B/C/D"])
                    self.assertAlmostEqual(0.1 * i, item["A/B/C/E"])

            with self.subTest("Test instrument data"):
                for i, (pid, item) in enumerate(data):
                    np.testing.assert_array_equal(np.ones(s1), item["H/I/J/K"])
                    np.testing.assert_array_equal(np.ones(s2), item["H/I/J/L"])

            with self.subTest("Test reading a single control data channel"):
                item = data.channel('A/B/C/D')
                with self.assertRaises(KeyError):
                    item[2]
                self.assertEqual(10., item[3])
                self.assertEqual(10., item.from_index(1))
                item_array = item.numpy()
                self.assertEqual(np.float32, item_array.dtype)
                np.testing.assert_array_equal(10. * np.arange(10), item_array)

            with self.subTest("Test reading a single instrument data channel"):
                item = data.channel('H/I/J/L')
                np.testing.assert_array_equal(np.ones(s2), item[1])
                item_array = item.numpy()
                self.assertEqual(np.float32, item_array.dtype)
                np.testing.assert_array_equal(np.ones((10, *s2)), item_array)
