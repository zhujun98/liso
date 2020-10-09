import unittest
import tempfile
import os

import numpy as np
import pandas as pd
import h5py

from liso import Phasespace
from liso.io import SimWriter, open_run, open_sim
from liso.io.reader import ExpDataCollection, SimDataCollection


class TestReader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # prepare a simulation file
        cls._n_sims = 10
        cls._sim_file = tempfile.NamedTemporaryFile(suffix=".hdf5")
        writer = SimWriter(cls._n_sims, 100, cls._sim_file.name)

        for i in range(cls._n_sims):
            ps1 = Phasespace(
                pd.DataFrame(np.ones((100, 7)) * i,
                             columns=['x', 'px', 'y', 'py', 'z', 'pz', 't']), 1.0)

            ps2 = Phasespace(
                pd.DataFrame(np.ones((100, 7)) * i * 10,
                             columns=['x', 'px', 'y', 'py', 'z', 'pz', 't']), 1.0)

            writer.write(i,
                         {'gun/gun_gradient': 10 * i, 'gun/gun_phase': 20 * i},
                         {'gun/out1': ps1, 'gun/out2': ps2})

        # prepare an experimental file
        cls._n_pulses = 10
        cls._pulse_ids = np.arange(1, 2 * cls._n_pulses, 2)
        cls._exp_file = tempfile.NamedTemporaryFile(suffix=".hdf5")
        cls._exp_control_data = {
            'XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/PHASE.SAMPLE':
                np.arange(cls._n_pulses).astype(np.float32),
            'XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/AMPL.SAMPLE':
                np.arange(cls._n_pulses).astype(np.float32) / 10.
        }
        cls._image_shape = (4, 25)
        cls._exp_detector_data = {
            'XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ':
                np.ones(cls._n_pulses * 100).astype(np.uint16).reshape(
                    (cls._n_pulses, *cls._image_shape))
        }
        with h5py.File(cls._exp_file.name, 'w') as fp_h5:
            fp_h5.create_dataset("INDEX/pulseId", data=cls._pulse_ids)

            fp_h5.create_dataset(
                "METADATA/controlChannels",
                dtype=h5py.string_dtype(),
                data=[ch.encode("utf-8") for ch in cls._exp_control_data])
            for ch, data_ in cls._exp_control_data.items():
                fp_h5.create_dataset(f"CONTROL/{ch}", data=data_)

            fp_h5.create_dataset(
                "METADATA/detectorChannels",
                dtype=h5py.string_dtype(),
                data=[ch.encode("utf-8") for ch in cls._exp_detector_data])
            for ch, data_ in cls._exp_detector_data.items():
                fp_h5.create_dataset(f"DETECTOR/{ch}", data=data_)

    @classmethod
    def tearDownClass(cls):
        cls._sim_file.close()
        cls._exp_file.close()

        assert(not os.path.isfile(cls._sim_file.name))
        assert(not os.path.isfile(cls._exp_file.name))

    def testOpenSim(self):
        data = open_sim(self._sim_file.name)
        data.info()
        self.assertIsInstance(data, SimDataCollection)
        self.assertSetEqual({'gun/gun_gradient', 'gun/gun_phase'},
                            data.control_channels)
        self.assertSetEqual({'gun/out1', 'gun/out2'}, data.phasespace_channels)
        self.assertListEqual([i+1 for i in range(self._n_sims)], data.sim_ids)

        # test "get_controls" method
        control_data = data.get_controls()
        self.assertEqual(self._n_sims, len(control_data))
        self.assertEqual(['gun/gun_gradient', 'gun/gun_phase'],
                         control_data.columns.tolist())
        np.testing.assert_array_equal(10 * np.arange(10),
                                      control_data['gun/gun_gradient'])
        np.testing.assert_array_equal(20 * np.arange(10),
                                      control_data['gun/gun_phase'])
        self.assertListEqual([i+1 for i in range(self._n_sims)],
                             control_data.index.tolist())

        # test access data by simId
        with self.assertRaises(KeyError):
            data[0]

        for sid, sim in data:
            self.assertSetEqual({
                'gun/gun_phase', 'gun/gun_gradient', 'gun/out1', 'gun/out2'
            }, set(sim.keys()))
            i = sid - 1
            self.assertEqual(20 * i, sim['gun/gun_phase'])
            self.assertEqual(10 * i, sim['gun/gun_gradient'])
            np.testing.assert_array_equal(np.ones(100) * i, sim['gun/out1']['x'])
            np.testing.assert_array_equal(np.ones(100) * 10 * i, sim['gun/out2']['y'])

        # test access data by index

        sim_id, sim = data.from_index(1)
        self.assertEqual(2, sim_id)
        self.assertSetEqual({
            'gun/gun_phase', 'gun/gun_gradient', 'gun/out1', 'gun/out2'
        }, set(sim.keys()))
        self.assertEqual(20 * 1, sim['gun/gun_phase'])
        self.assertEqual(10 * 1, sim['gun/gun_gradient'])
        np.testing.assert_array_equal(np.ones(100) * 1, sim['gun/out1']['x'])
        np.testing.assert_array_equal(np.ones(100) * 10 * 1, sim['gun/out2']['y'])

    def testChannelData(self):
        # simulation
        data = open_sim(self._sim_file.name)

        with self.assertRaisesRegex(KeyError, 'No data was found for channel'):
            data.channel('gun/random')

        item = data.channel('gun/gun_phase')
        with self.assertRaises(KeyError):
            item[0]
        with self.assertRaises(KeyError):
            item[11]
        self.assertEqual(9 * 20, item[10])
        self.assertEqual(9 * 20, item.from_index(9))
        item_array = item.numpy()
        self.assertEqual(np.float64, item_array.dtype)
        np.testing.assert_array_equal(20 * np.arange(10), item_array)

        # simulation - phasespace data
        item = data.channel('gun/out1')
        self.assertEqual(np.float64, item_array.dtype)
        np.testing.assert_array_equal(2 * np.ones((7, 100)), item[3])

        item = data.channel('gun/out1', 't')
        np.testing.assert_array_equal(3 * np.ones((1, 100)), item[4])

        item = data.channel('gun/out1', ['x', 'y'])
        np.testing.assert_array_equal(9 * np.ones((2, 100)), item[10])

        with self.assertRaisesRegex(ValueError, "not a valid phasespace column"):
            data.channel('gun/out1', 'a')

        with self.assertRaisesRegex(ValueError, "not a valid phasespace column"):
            data.channel('gun/out1', ['x', 'a'])

        # experiments
        data = open_run(self._exp_file.name)

        item = data.channel('XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/PHASE.SAMPLE')
        with self.assertRaises(KeyError):
            item[2]
        self.assertEqual(1., item[3])
        self.assertEqual(1., item.from_index(1))
        item_array = item.numpy()
        self.assertEqual(np.float32, item_array.dtype)
        np.testing.assert_array_equal(np.arange(10), item_array)

        item = data.channel('XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ')
        np.testing.assert_array_equal(np.ones(self._image_shape), item[1])
        item_array = item.numpy()
        self.assertEqual(np.uint16, item_array.dtype)
        np.testing.assert_array_equal(np.ones((10, *self._image_shape)), item_array)

    def testOpenRun(self):
        data = open_run(self._exp_file.name)
        self.assertIsInstance(data, ExpDataCollection)

        controls = data.get_controls()
        self.assertListEqual(list(self._exp_control_data.keys()), controls.columns.tolist())
        np.testing.assert_array_equal(self._pulse_ids, controls.index.to_numpy())
        for ch, v in self._exp_control_data.items():
            np.testing.assert_array_equal(controls[ch], v)

        for i, (pid, item) in enumerate(data):
            self.assertSetEqual({
                'XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/PHASE.SAMPLE',
                'XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/AMPL.SAMPLE',
                'XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ'
            }, set(item.keys()))

            self.assertAlmostEqual(
                i, item['XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/PHASE.SAMPLE'])
            self.assertAlmostEqual(
                i / 10., item['XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/AMPL.SAMPLE'])
            np.testing.assert_array_equal(
                np.ones(100).reshape(4, 25),
                item['XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ'])
