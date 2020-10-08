import unittest
import tempfile

import numpy as np
import pandas as pd
import h5py

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
                             {'gun/gun_gradient': 10 * i, 'gun/gun_phase': 20 * i},
                             {'gun/out1': ps1, 'gun/out2': ps2})

            data = open_sim(fp.name)
            data.info()
            self.assertIsInstance(data, SimDataCollection)
            self.assertSetEqual({'gun/gun_gradient', 'gun/gun_phase'},
                                data.control_channels)
            self.assertSetEqual({'gun/out1', 'gun/out2'}, data.phasespace_channels)
            self.assertListEqual([i+1 for i in range(n_sims)], data.sim_ids)

            # test "get_controls" method
            control_data = data.get_controls()
            self.assertEqual(n_sims, len(control_data))
            self.assertEqual(['gun/gun_gradient', 'gun/gun_phase'],
                             control_data.columns.tolist())
            np.testing.assert_array_equal(10 * np.arange(10),
                                          control_data['gun/gun_gradient'])
            np.testing.assert_array_equal(20 * np.arange(10),
                                          control_data['gun/gun_phase'])
            self.assertListEqual([i+1 for i in range(n_sims)],
                                 control_data.index.tolist())

            # test access data by simId
            with self.assertRaises(IndexError):
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

            sim_id, sim = data.iloc(1)
            self.assertEqual(2, sim_id)
            self.assertSetEqual({
                'gun/gun_phase', 'gun/gun_gradient', 'gun/out1', 'gun/out2'
            }, set(sim.keys()))
            self.assertEqual(20 * 1, sim['gun/gun_phase'])
            self.assertEqual(10 * 1, sim['gun/gun_gradient'])
            np.testing.assert_array_equal(np.ones(100) * 1, sim['gun/out1']['x'])
            np.testing.assert_array_equal(np.ones(100) * 10 * 1, sim['gun/out2']['y'])

    def testOpenRun(self):
        with tempfile.NamedTemporaryFile(suffix=".hdf5") as fp:
            with h5py.File(fp.name, 'w') as fp_h5:
                n_pulses = 10
                pulse_ids = np.arange(1, 2*n_pulses, 2)

                control_data = {
                    'XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/PHASE.SAMPLE':
                        np.arange(n_pulses).astype(np.float32),
                    'XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/AMPL.SAMPLE':
                        np.arange(n_pulses).astype(np.float32) / 10.
                }

                detector_data = {
                    'XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ':
                    np.ones(n_pulses * 100).reshape((n_pulses, 4, 25))
                }

                fp_h5.create_dataset("INDEX/pulseId", data=pulse_ids)

                fp_h5.create_dataset(
                    "METADATA/controlChannels",
                    dtype=h5py.string_dtype(),
                    data=[ch.encode("utf-8") for ch in control_data])
                for ch, data_ in control_data.items():
                    fp_h5.create_dataset(f"CONTROL/{ch}", data=data_)

                fp_h5.create_dataset(
                    "METADATA/detectorChannels",
                    dtype=h5py.string_dtype(),
                    data=[ch.encode("utf-8") for ch in detector_data])
                for ch, data_ in detector_data.items():
                    fp_h5.create_dataset(f"DETECTOR/{ch}", data=data_)

            data = open_run(fp.name)
            self.assertIsInstance(data, ExpDataCollection)

            controls = data.get_controls()
            self.assertListEqual(list(control_data.keys()), controls.columns.tolist())
            np.testing.assert_array_equal(pulse_ids, controls.index.to_numpy())
            for ch, v in control_data.items():
                np.testing.assert_array_equal(controls[ch], v)

            for i, (pid, item) in enumerate(data):
                self.assertSetEqual({
                    'XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/PHASE.SAMPLE',
                    'XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/AMPL.SAMPLE',
                    'XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ'
                }, set(item.keys()))

                self.assertAlmostEqual(i, item['XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/PHASE.SAMPLE'])
                self.assertAlmostEqual(i / 10., item['XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/AMPL.SAMPLE'])
                np.testing.assert_array_equal(
                    np.ones(100).reshape(4, 25),
                    item['XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ'])
