import unittest
import tempfile
import pathlib
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
    assert 0 < (datetime.now() - create_date).total_seconds() < 1.0
    update_date = datetime.fromisoformat(fp['METADATA/updateDate'][()])
    assert 0 < (datetime.now() - update_date).total_seconds() < 1.0
    assert (update_date - create_date).total_seconds() > 0


class TestSimWriter(unittest.TestCase):
    def setUp(self):
        self._n_particles = 21

        self._chunk_size = 20
        self._file_size = 50

        self._sim_ids_gt = np.arange(2 * self._file_size + self._chunk_size + 9) + self._file_size
        np.random.shuffle(self._sim_ids_gt)  # sim_id does not come in sequence

    def testWrite(self):
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
                    "gun/out": {
                        "macroparticles": n_particles,
                        "type": "phasespace"
                    }
                }
            )

            with self.subTest("Test invalid input in writer"):
                with self.assertRaisesRegex(ValueError, "cannot be smaller than chunk_size"):
                    SimWriter(tmp_dir, schema=schema,
                              chunk_size=501, max_events_per_file=500)
                with self.assertRaisesRegex(ValueError, "group must be an integer"):
                    SimWriter(tmp_dir, schema=schema, group=0)

            with SimWriter(tmp_dir,
                           schema=schema,
                           chunk_size=chunk_size,
                           max_events_per_file=file_size) as writer:
                ps = Phasespace(
                    pd.DataFrame(np.ones((self._n_particles, 7)),
                                 columns=['x', 'px', 'y', 'py', 'z', 'pz',
                                          't']), 1.0)

                for i, sid in enumerate(sim_ids_gt):
                    writer.write(sid,
                                 {'gun/gun_gradient': 10 * i, 'gun/gun_phase': 0.1 * i},
                                 {'gun/out': ps})

            path = pathlib.Path(tmp_dir)
            files = sorted([f.name for f in path.iterdir()])
            self.assertListEqual(
                [f"SIM-G01-S00000{i}.hdf5" for i in range(3)], files
            )

            with self.subTest("Initialize writer when hdf5 file already exits"):
                with SimWriter(tmp_dir, schema=schema) as writer:
                    with self.assertRaises(OSError):
                        writer.write(0,
                                     {'gun/gun_gradient': 0, 'gun/gun_phase': 0},
                                     {'gun/out': ps})

            with self.subTest("Test data in the file"):
                self._check_data_files(path, files)

        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.subTest("Writer with particle loss"):
                with SimWriter(tmp_dir, schema=schema) as writer:
                    ps = Phasespace(pd.DataFrame(
                        np.ones((n_particles + 1, 7)),
                        columns=['x', 'px', 'y', 'py', 'z', 'pz', 't']), 1.0)
                    writer.write(0,
                                {'gun/gun_gradient': 1, 'gun/gun_phase': 2},
                                {'gun/out': ps})

                    for col in ['x', 'y', 'z', 'px', 'py', 'pz', 't']:
                        self.assertFalse(
                            np.any(writer._fp[f'PHASESPACE/{col.upper()}/gun/out'][0, ...]))

    def _check_data_files(self, path, files):
        chunk_size = self._chunk_size
        file_size = self._file_size
        n_particles = self._n_particles
        sim_ids_gt = self._sim_ids_gt

        for i, file in enumerate(files):
            with h5py.File(path.joinpath(file), 'r') as fp:
                _check_create_update_date(fp)

                self.assertSetEqual({'gun/gun_gradient', 'gun/gun_phase'},
                                    set(fp['METADATA/controlChannel']))
                self.assertSetEqual({'gun/out'},
                                    set(fp['METADATA/phasespaceChannel']))

                if i == 2:
                    np.testing.assert_array_equal(
                        sim_ids_gt[-(chunk_size + 9):], fp['INDEX/simId'][()])
                else:
                    np.testing.assert_array_equal(
                        sim_ids_gt[i * file_size:(i + 1) * file_size],
                        fp['INDEX/simId'][()])

                if i == 2:
                    np.testing.assert_array_equal(
                        np.concatenate((
                            10 * (np.arange(chunk_size + 9) + i * file_size),
                            np.zeros((chunk_size - 9)))),  # 2 * chunk_size - (chunk_size + 9)
                        fp['CONTROL/gun/gun_gradient'][()])
                    np.testing.assert_array_almost_equal(
                        np.concatenate((
                            0.1 * (np.arange(chunk_size + 9) + i * file_size),
                            np.zeros((chunk_size - 9)))),
                        fp['CONTROL/gun/gun_phase'][()])
                else:
                    np.testing.assert_array_equal(
                        10 * (np.arange(file_size) + i * file_size),
                        fp['CONTROL/gun/gun_gradient'][()])
                    np.testing.assert_array_almost_equal(
                        0.1 * (np.arange(file_size) + i * file_size),
                        fp['CONTROL/gun/gun_phase'][()])

                for col in ['x', 'y', 'z', 'px', 'py', 'pz', 't']:
                    if i == 2:
                        np.testing.assert_array_equal(
                            np.concatenate((
                                np.ones((chunk_size + 9, n_particles)),
                                np.zeros((chunk_size - 9, n_particles)))),
                            fp[f'PHASESPACE/{col.upper()}/gun/out'][()])
                    else:
                        np.testing.assert_array_equal(
                            np.ones((file_size, n_particles)),
                            fp[f'PHASESPACE/{col.upper()}/gun/out'][()])


class TestExpWriter(unittest.TestCase):
    def setUp(self):
        m = EuXFELInterface()
        m.add_control_channel(dc.FLOAT64, "XFEL.A/B/C/D")
        m.add_control_channel(dc.FLOAT32, "XFEL.A/B/C/E")

        self._s1 = (4, 4)
        self._s2 = (5, 6)
        m.add_diagnostic_channel(dc.IMAGE, "XFEL.H/I/J/K", shape=self._s1, dtype='uint16')
        m.add_diagnostic_channel(dc.IMAGE, "XFEL.H/I/J/L", shape=self._s2, dtype='float32')
        self._schema = m.schema

        self._orig_image_chunk = ExpWriter._IMAGE_CHUNK
        ExpWriter._IMAGE_CHUNK = (2, 3)

        self._chunk_size = 30
        self._file_size = 50

        self._pulse_ids_gt = np.arange(2 * self._file_size + self._chunk_size + 9) + self._file_size

    def tearDown(self):
        ExpWriter._IMAGE_CHUNK = self._orig_image_chunk

    def testWrite(self):
        chunk_size = self._chunk_size
        file_size = self._file_size
        s1, s2 = self._s1, self._s2
        pulse_ids_gt = self._pulse_ids_gt

        with tempfile.TemporaryDirectory() as tmp_dir:
            with ExpWriter(tmp_dir,
                           schema=self._schema,
                           group=11,
                           chunk_size=chunk_size,
                           max_events_per_file=file_size) as writer:

                for i, pid in enumerate(pulse_ids_gt):
                    writer.write(
                        pid,
                        {"XFEL.A/B/C/D": 10 * i, "XFEL.A/B/C/E": 0.1 * i},
                        {"XFEL.H/I/J/K": np.ones(s1, dtype=np.uint16),
                         "XFEL.H/I/J/L": np.ones(s2, dtype=np.float32)}
                    )

            path = pathlib.Path(tmp_dir)
            files = sorted([f.name for f in path.iterdir()])
            self.assertListEqual(
                [f"RAW-{path.name.upper()}-G11-S00000{i}.hdf5" for i in range(3)], files
            )
            with self.subTest("Test data in the file"):
                self._check_data_files(path, files)

    def _check_data_files(self, path, files):
        chunk_size = self._chunk_size
        file_size = self._file_size
        s1, s2 = self._s1, self._s2
        pulse_ids_gt = self._pulse_ids_gt

        for i, file in enumerate(files):
            with h5py.File(path.joinpath(file), 'r') as fp:
                _check_create_update_date(fp)

                self.assertSetEqual(
                    {"XFEL.A/B/C/D", "XFEL.A/B/C/E"}, set(fp['METADATA/controlChannel']))
                self.assertSetEqual(
                    {"XFEL.H/I/J/K", "XFEL.H/I/J/L"}, set(fp['METADATA/diagnosticChannel']))

                if i == 2:
                    np.testing.assert_array_equal(
                        pulse_ids_gt[-(chunk_size + 9):], fp['INDEX/pulseId'][()])
                else:
                    np.testing.assert_array_equal(
                        pulse_ids_gt[i * file_size:(i + 1) * file_size],
                        fp['INDEX/pulseId'][()])

                self.assertEqual(np.float64, fp['CONTROL/XFEL.A/B/C/D'].dtype)
                self.assertEqual(np.float32, fp['CONTROL/XFEL.A/B/C/E'].dtype)
                if i == 2:
                    np.testing.assert_array_equal(
                        np.concatenate((
                            10 * (np.arange(chunk_size + 9) + i * file_size),
                            np.zeros((file_size - chunk_size - 9)))),
                        fp['CONTROL/XFEL.A/B/C/D'][()])
                    np.testing.assert_array_almost_equal(
                        np.concatenate((
                            0.1 * (np.arange(chunk_size + 9) + i * file_size),
                            np.zeros((file_size - chunk_size - 9)))),
                        fp['CONTROL/XFEL.A/B/C/E'][()])
                else:
                    np.testing.assert_array_equal(
                        10 * (np.arange(file_size) + i * file_size),
                        fp['CONTROL/XFEL.A/B/C/D'][()])
                    np.testing.assert_array_almost_equal(
                        0.1 * (np.arange(file_size) + i * file_size),
                        fp['CONTROL/XFEL.A/B/C/E'][()])

                self.assertEqual((file_size, *s1), fp['DIAGNOSTIC/XFEL.H/I/J/K'].shape)
                self.assertEqual((file_size, *s2), fp['DIAGNOSTIC/XFEL.H/I/J/L'].shape)
                self.assertEqual(np.uint16, fp['DIAGNOSTIC/XFEL.H/I/J/K'].dtype)
                self.assertEqual(np.float32, fp['DIAGNOSTIC/XFEL.H/I/J/L'].dtype)

                if i == 2:
                    np.testing.assert_array_equal(
                        np.concatenate((
                            np.ones((chunk_size + 9, *s1)),
                            np.zeros((file_size - chunk_size - 9, *s1)))),
                        fp['DIAGNOSTIC/XFEL.H/I/J/K'][()])
                    np.testing.assert_array_equal(
                        np.concatenate((
                            np.ones((chunk_size + 9, *s2)),
                            np.zeros((file_size - chunk_size - 9, *s2)))),
                        fp['DIAGNOSTIC/XFEL.H/I/J/L'][()])
                else:
                    np.testing.assert_array_equal(
                        np.ones((file_size, *s1)), fp['DIAGNOSTIC/XFEL.H/I/J/K'][()])
                    np.testing.assert_array_equal(
                        np.ones((file_size, *s2)), fp['DIAGNOSTIC/XFEL.H/I/J/L'][()])
