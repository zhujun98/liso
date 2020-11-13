import unittest
import tempfile
import pathlib

import numpy as np
import pandas as pd

from liso import EuXFELInterface, Phasespace
from liso.experiment import doocs_channels as dc
from liso.io import ExpWriter, SimWriter, open_run, open_sim
from liso.io.reader import ExpDataCollection, SimDataCollection


class TestSimReader(unittest.TestCase):
    def setUp(self):
        self._n_particles = 20
        self._ps = Phasespace(
            pd.DataFrame(np.ones((self._n_particles, 7)),
                         columns=['x', 'px', 'y', 'py', 'z', 'pz', 't']), 1.0)

        self._chunk_size = 25
        self._file_size = 50

        self._sim_ids_gt = np.arange(2 * self._file_size + self._chunk_size + 9) + self._file_size
        np.random.shuffle(self._sim_ids_gt)  # sim_id does not come in sequence

    def testOpenSim(self):
        chunk_size = self._chunk_size
        file_size = self._file_size
        n_particles = self._n_particles
        sim_ids_gt = self._sim_ids_gt

        with tempfile.TemporaryDirectory() as tmp_dir:
            schema = (
                {
                    "gun/gun_gradient": {"type": "<f4"},
                    "gun/gun_phase": {"type": "<f4"}
                },
                {
                    "gun/out1": {
                        "macroparticles": n_particles,
                        "type": "phasespace"
                    },
                    "gun/out2": {
                        "macroparticles": n_particles,
                        "type": "phasespace"
                    }
                }
            )

            with SimWriter(tmp_dir,
                           schema=schema,
                           chunk_size=chunk_size,
                           max_events_per_file=file_size) as writer:

                for i, sid in enumerate(sim_ids_gt):
                    ps1 = Phasespace(
                        pd.DataFrame(np.ones((n_particles, 7)) * i,
                                     columns=['x', 'px', 'y', 'py', 'z', 'pz', 't']), 1.0)

                    ps2 = Phasespace(
                        pd.DataFrame(np.ones((n_particles, 7)) * i * 0.1,
                                     columns=['x', 'px', 'y', 'py', 'z', 'pz', 't']), 1.0)

                    writer.write(sid,
                                 {'gun/gun_gradient': 10 * i, 'gun/gun_phase': 20 * i},
                                 {'gun/out1': ps1, 'gun/out2': ps2})

            path = pathlib.Path(tmp_dir)
            files = sorted([f.name for f in path.iterdir()])

            with self.subTest("Test opening a single file"):
                for i, file in enumerate(files):
                    data = open_sim(path.joinpath(file))
                    self.assertIsInstance(data, SimDataCollection)
                    data.info()
                    self._check_sim_metadata(data, i)
                    self._check_sim_get_control(data, i)
                    self._check_sim_iterate_over_data(data, i)
                    self._check_sim_access_data_by_index(data, i)

            with self.subTest("Test opening a folder"):
                data = open_sim(tmp_dir)
                self._check_sim_metadata(data)
                self._check_sim_get_control(data)
                self._check_sim_iterate_over_data(data)
                self._check_sim_access_data_by_index(data)

            with self.subTest("Test control channel data"):
                with self.assertRaisesRegex(KeyError, 'No data was found for channel'):
                    data.channel('gun/random')

                item = data.channel('gun/gun_phase')
                with self.assertRaises(KeyError):
                    data[min(self._sim_ids_gt) - 1]
                with self.assertRaises(KeyError):
                    data[max(self._sim_ids_gt) + 1]

                idx = 9
                self.assertEqual(idx * 20, item[self._sim_ids_gt[idx]])
                self.assertEqual(idx * 20, item.from_index(idx))
                self.assertEqual(np.float32, item.numpy().dtype)
                np.testing.assert_array_equal(
                    20 * np.arange(len(self._sim_ids_gt)), item.numpy())

            with self.subTest("Test phasespace channel data"):
                item = data.channel('gun/out1')
                self.assertEqual(np.float64, item.numpy().dtype)
                np.testing.assert_array_equal(
                    3 * np.ones((7, n_particles)), item.from_index(3))

                idx = 59
                item = data.channel('gun/out1', 't')
                np.testing.assert_array_equal(
                    idx * np.ones((1, self._n_particles)), item[self._sim_ids_gt[idx]])
                item = data.channel('gun/out1', ['x', 'y'])
                np.testing.assert_array_equal(
                    idx * np.ones((2, self._n_particles)), item[self._sim_ids_gt[idx]])

                with self.assertRaisesRegex(ValueError, "not a valid phasespace column"):
                    data.channel('gun/out1', 'a')

                with self.assertRaisesRegex(ValueError, "not a valid phasespace column"):
                    data.channel('gun/out1', ['x', 'a'])

    def _check_sim_metadata(self, data, i=None):
        file_size = self._file_size
        chunk_size = self._chunk_size
        self.assertSetEqual({'gun/gun_gradient', 'gun/gun_phase'},
                            data.control_channels)
        self.assertSetEqual({'gun/out1', 'gun/out2'}, data.phasespace_channels)
        if i is None:
            np.testing.assert_array_equal(self._sim_ids_gt, data.sim_ids)
        elif i == 2:
            np.testing.assert_array_equal(
                self._sim_ids_gt[-(chunk_size + 9):], data.sim_ids)
        else:
            np.testing.assert_array_equal(
                self._sim_ids_gt[i*file_size:(i+1)*file_size], data.sim_ids)

    def _check_sim_get_control(self, data, i=None):
        file_size = self._file_size
        chunk_size = self._chunk_size

        control_data = data.get_controls()
        self.assertEqual(['gun/gun_gradient', 'gun/gun_phase'],
                         control_data.columns.tolist())

        if i is None:
            self.assertEqual(len(self._sim_ids_gt), len(control_data))
            np.testing.assert_array_equal(10 * np.arange(len(self._sim_ids_gt)),
                                          control_data['gun/gun_gradient'])
            np.testing.assert_array_equal(20 * np.arange(len(self._sim_ids_gt)),
                                          control_data['gun/gun_phase'])
            np.testing.assert_array_equal(self._sim_ids_gt, control_data.index)

            # test sorted
            sorted_control_data = data.get_controls(sorted=True)
            np.testing.assert_array_equal(sorted(self._sim_ids_gt),
                                          sorted_control_data.index)
        elif i == 2:
            self.assertEqual(self._chunk_size + 9, len(control_data))
            indices = (2 * file_size, 2 * file_size + chunk_size + 9)
            np.testing.assert_array_equal(10 * np.arange(*indices),
                                          control_data['gun/gun_gradient'])
            np.testing.assert_array_equal(20 * np.arange(*indices),
                                          control_data['gun/gun_phase'])
            np.testing.assert_array_equal(
                self._sim_ids_gt[indices[0]:indices[1]], control_data.index)
        else:
            self.assertEqual(self._file_size, len(control_data))
            indices = (i * file_size, (i + 1) * file_size)
            np.testing.assert_array_equal(10 * np.arange(*indices),
                                          control_data['gun/gun_gradient'])
            np.testing.assert_array_equal(20 * np.arange(*indices),
                                          control_data['gun/gun_phase'])
            np.testing.assert_array_equal(
                self._sim_ids_gt[indices[0]:indices[1]], control_data.index)

    def _check_sim_iterate_over_data(self, data, i=None):
        with self.assertRaises(KeyError):
            data[min(self._sim_ids_gt) - 1]
        with self.assertRaises(KeyError):
            data[max(self._sim_ids_gt) + 1]

        for idx, (sid, sim) in enumerate(data):
            if i is not None:
                idx += i * self._file_size
            self.assertSetEqual({
                'gun/gun_phase', 'gun/gun_gradient', 'gun/out1', 'gun/out2'
            }, set(sim.keys()))
            self.assertEqual(20 * idx, sim['gun/gun_phase'])
            self.assertEqual(10 * idx, sim['gun/gun_gradient'])
            np.testing.assert_array_equal(
                np.ones(self._n_particles) * idx, sim['gun/out1']['x'])
            np.testing.assert_array_equal(
                np.ones(self._n_particles) * idx * 0.1, sim['gun/out2']['y'])

    def _check_sim_access_data_by_index(self, data, i=None):
        if i is None:
            idx = np.random.randint(0, len(self._sim_ids_gt))
            sim_id, sim = data.from_index(idx)
        elif i == 2:
            idx = np.random.randint(0, self._chunk_size + 9)
            sim_id, sim = data.from_index(idx)
            idx += 2 * self._file_size
        else:
            idx = np.random.randint(0, self._file_size)
            sim_id, sim = data.from_index(idx)
            idx += i * self._file_size

        self.assertEqual(self._sim_ids_gt[idx], sim_id)

        self.assertSetEqual({
            'gun/gun_phase', 'gun/gun_gradient', 'gun/out1', 'gun/out2'
        }, set(sim.keys()))
        self.assertEqual(20 * idx, sim['gun/gun_phase'])
        self.assertEqual(10 * idx, sim['gun/gun_gradient'])
        np.testing.assert_array_equal(
            np.ones(self._n_particles) * idx, sim['gun/out1']['x'])
        np.testing.assert_array_equal(
            np.ones(self._n_particles) * idx * 0.1, sim['gun/out2']['y'])


class TestExpReader(unittest.TestCase):
    def setUp(self):
        m = EuXFELInterface()
        m.add_control_channel(dc.FLOAT32, "XFEL.A/B/C/D")
        m.add_control_channel(dc.FLOAT32, "XFEL.A/B/C/E")
        m.add_control_channel(dc.BOOL, "XFEL.A/B/C/F")
        self._s1 = (3, 4)
        self._s2 = (6, 5)
        m.add_diagnostic_channel(dc.IMAGE, "XFEL.H/I/J/K", shape=self._s1, dtype='uint16')
        m.add_diagnostic_channel(dc.IMAGE, "XFEL.H/I/J/L", shape=self._s2, dtype='float32')
        self._schema = m.schema

        self._orig_image_chunk = ExpWriter._IMAGE_CHUNK
        ExpWriter._IMAGE_CHUNK = (3, 2)

        self._chunk_size = 30
        self._file_size = 50

        self._pulse_ids_gt = np.arange(2 * self._file_size + self._chunk_size + 9) + self._file_size

    def tearDown(self):
        ExpWriter._IMAGE_CHUNK = self._orig_image_chunk

    def testOpenRun(self):
        chunk_size = self._chunk_size
        file_size = self._file_size
        s1, s2 = self._s1, self._s2
        pulse_ids_gt = self._pulse_ids_gt

        with tempfile.TemporaryDirectory() as tmp_dir:
            with ExpWriter(tmp_dir,
                           schema=self._schema,
                           chunk_size=chunk_size,
                           max_events_per_file=file_size) as writer:

                for i, pid in enumerate(pulse_ids_gt):
                    writer.write(
                        pid,
                        {"XFEL.A/B/C/D": 10 * i, "XFEL.A/B/C/E": 0.1 * i},
                        {"XFEL.H/I/J/K": np.ones(s1, dtype=np.uint16),
                         "XFEL.H/I/J/L": np.ones(s2, dtype=np.float32)}
                    )

            data = open_run(tmp_dir)
            self.assertIsInstance(data, ExpDataCollection)
            data.info()

            with self.subTest("Test control data"):
                controls = data.get_controls()
                self.assertListEqual(["XFEL.A/B/C/D", "XFEL.A/B/C/E", "XFEL.A/B/C/F"],
                                     controls.columns.tolist())

                np.testing.assert_array_equal(
                    pulse_ids_gt, controls.index.to_numpy())
                np.testing.assert_array_equal(
                    10. * np.arange(len(pulse_ids_gt)), controls["XFEL.A/B/C/D"])
                np.testing.assert_array_almost_equal(
                    .1 * np.arange(len(pulse_ids_gt)), controls["XFEL.A/B/C/E"])
                np.testing.assert_array_equal(
                    np.zeros(len(pulse_ids_gt)).astype(bool), controls["XFEL.A/B/C/F"])

                for i, (pid, item) in enumerate(data):
                    self.assertSetEqual({
                        "XFEL.A/B/C/D", "XFEL.A/B/C/E", "XFEL.A/B/C/F",
                        "XFEL.H/I/J/K", "XFEL.H/I/J/L"
                    }, set(item.keys()))

                    self.assertAlmostEqual(10. * i, item["XFEL.A/B/C/D"])
                    self.assertAlmostEqual(0.1 * i, item["XFEL.A/B/C/E"], places=4)

            with self.subTest("Test diagnostic data"):
                for i, (pid, item) in enumerate(data):
                    np.testing.assert_array_equal(np.ones(s1), item["XFEL.H/I/J/K"])
                    np.testing.assert_array_equal(np.ones(s2), item["XFEL.H/I/J/L"])

            with self.subTest("Test reading a single control data channel"):
                item = data.channel('XFEL.A/B/C/D')
                with self.assertRaises(KeyError):
                    item[2]
                self.assertEqual(30., item[pulse_ids_gt[3]])
                self.assertEqual(10., item.from_index(1))
                item_array = item.numpy()
                self.assertEqual(np.float32, item_array.dtype)
                np.testing.assert_array_equal(
                    10. * np.arange(len(pulse_ids_gt)), item_array)

            with self.subTest("Test reading a single diagnostic data channel"):
                item = data.channel('XFEL.H/I/J/L')
                np.testing.assert_array_equal(np.ones(s2), item[pulse_ids_gt[1]])
                item_array = item.numpy()
                self.assertEqual(np.float32, item_array.dtype)
                np.testing.assert_array_equal(
                    np.ones((len(pulse_ids_gt), *s2)), item_array)
