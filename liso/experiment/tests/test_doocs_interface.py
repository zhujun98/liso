import asyncio
from pathlib import Path
import unittest
from unittest.mock import patch
import tempfile

import numpy as np

from liso import EuXFELInterface
from liso import doocs_channels as dc
from liso.exceptions import LisoRuntimeError
from liso.experiment.doocs_interface import DoocsException, PyDoocsException
from liso.io import ExpWriter, open_run
from liso.logging import logger

from . import DoocsDataGenerator as ddgen


_INITIAL_PID = 1000


def _side_effect_read(dataset, address):
    data = dataset[address].copy()
    if data['macropulse'] >= _INITIAL_PID:
        dataset[address]['macropulse'] += 1
    return data


class TestDoocsInterface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._orig_image_chunk = ExpWriter._IMAGE_CHUNK
        ExpWriter._IMAGE_CHUNK = (3, 2)

    @classmethod
    def tearDownClass(cls):
        ExpWriter._IMAGE_CHUNK = cls._orig_image_chunk

    def setUp(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        logger.setLevel("ERROR")

        cfg = {
            "timeout.correlation": 0.04,
            "interval.read.retry": 0.002
        }
        m = EuXFELInterface(cfg)
        m._validation_prob = 1.
        m.add_control_channel("XFEL.A/B/C/D", dc.FLOAT, write_address="XFEL.A/B/C/d")
        m.add_control_channel("XFEL.A/B/C/E")
        m.add_diagnostic_channel("XFEL.H/I/J/K", dc.IMAGE,
                                 shape=(4, 4), dtype="uint16", non_event=True)
        m.add_diagnostic_channel("XFEL.H/I/J/L", dc.IMAGE,
                                 shape=(5, 6), dtype="float32")
        self._machine = m

        self._dataset = {
            "XFEL.A/B/C/D": ddgen.scalar(
                10., m._channels["XFEL.A/B/C/D"].value_schema(), pid=_INITIAL_PID),
            "XFEL.A/B/C/E": ddgen.scalar(
                100., m._channels["XFEL.A/B/C/E"].value_schema(), pid=_INITIAL_PID),
            "XFEL.H/I/J/K": ddgen.image(
                m._channels["XFEL.H/I/J/K"].value_schema(), pid=0),  # non-event warning
            "XFEL.H/I/J/L": ddgen.image(
                m._channels["XFEL.H/I/J/L"].value_schema(), pid=_INITIAL_PID)
        }

    def tearDown(self):
        asyncio.set_event_loop(None)
        self._loop.close()

    def testChannelManipulation(self):
        m = self._machine

        self.assertSetEqual({"XFEL.A/B/C/D", 'XFEL.A/B/C/E'}, m.control_channels)
        self.assertSetEqual({"XFEL.A/B/C/d", 'XFEL.A/B/C/E'}, set(m._control_write.values()))
        self.assertSetEqual({"XFEL.H/I/J/K", 'XFEL.H/I/J/L'}, m.diagnostic_channels)
        self.assertSetEqual({"XFEL.A/B/C/D", 'XFEL.A/B/C/E', "XFEL.H/I/J/K", 'XFEL.H/I/J/L'},
                            m.channels)
        self.assertSetEqual({"XFEL.A/B/C/D", 'XFEL.A/B/C/E', 'XFEL.H/I/J/L'}, m._event)
        self.assertSetEqual({"XFEL.H/I/J/K"}, m._non_event)

        with self.subTest("Add an existing channel"):
            with self.assertRaisesRegex(ValueError, "existing channel"):
                m.add_control_channel("XFEL.A/B/C/D", dc.IMAGE, shape=(2, 2), dtype="uint16")
            with self.assertRaisesRegex(ValueError, "existing channel"):
                m.add_diagnostic_channel("XFEL.H/I/J/K", dc.FLOAT)

        with self.subTest("Test schema"):
            m = self._machine
            schema = m.schema
            self.assertDictEqual(
                {'XFEL.A/B/C/D': {'default': 0.0, 'type': '<f4',
                                  'maximum': np.finfo(np.float32).max,
                                  'minimum': np.finfo(np.float32).min},
                 'XFEL.A/B/C/E': {'type': 'any'}},
                schema['control']
            )
            self.assertDictEqual(
                {'XFEL.H/I/J/K': {'dtype': '<u2', 'shape': (4, 4), 'type': 'NDArray'},
                 'XFEL.H/I/J/L': {'dtype': '<f4', 'shape': (5, 6), 'type': 'NDArray'}},
                schema['diagnostic']
            )

    @patch("liso.experiment.doocs_interface.pydoocs_write")
    def testWrite(self, patched_write):
        m = self._machine

        with self.assertRaisesRegex(KeyError, "not found in the control channels"):
            m.write(mapping={'XFEL.A/B/C/C': 1.})

        with self.assertRaisesRegex(LisoRuntimeError, "Failed to update 1/2 channels"):
            with self.assertLogs(level="ERROR") as cm:
                def _side_effect_write1(address, _):
                    if address == 'XFEL.A/B/C/E':
                        raise np.random.choice([PyDoocsException, DoocsException])
                patched_write.side_effect = _side_effect_write1
                m.write(mapping={
                    'XFEL.A/B/C/D': 1.,
                    'XFEL.A/B/C/E': 10.,
                })
        assert "Failed to write" in cm.output[0]

        with self.assertRaisesRegex(LisoRuntimeError, "Failed to update 1/2 channels"):
            with self.assertLogs(level="ERROR") as cm:
                def _side_effect_write2(address, _):
                    if address == 'XFEL.A/B/C/d':
                        raise np.random.choice([ValueError, RuntimeError])
                patched_write.side_effect = _side_effect_write2
                m.write(mapping={
                    'XFEL.A/B/C/D': 1.,
                    'XFEL.A/B/C/E': 10.,
                })
        assert "Unexpected exception" in cm.output[0]

    @patch("liso.experiment.doocs_interface.pydoocs_read")
    def testQuery(self, mocked_read):
        m = self._machine
        dataset = self._dataset
        mocked_read.side_effect = lambda x: _side_effect_read(dataset, x)

        with self.subTest("Normal"):
            data = m.query()
            assert mocked_read.call_count == 4
            assert len(data) == 4
            assert data['XFEL.A/B/C/D']['data'] == 10
            assert data['XFEL.A/B/C/E']['macropulse'] == 1000
            assert data['XFEL.H/I/J/K']['macropulse'] == 0
            assert data['XFEL.H/I/J/L']['type'] == 'IMAGE'

        with self.subTest("Raise when reading"):
            # raise happens to an event-based channel
            def _side_effect_read2(dataset, address):
                if address != "XFEL.H/I/J/K":
                    raise np.random.choice([PyDoocsException, DoocsException])
                return dataset[address]
            mocked_read.side_effect = lambda x: _side_effect_read2(dataset, x)
            data = m.query()
            assert len(data) == 4
            assert data['XFEL.A/B/C/D'] is None
            assert data['XFEL.A/B/C/E'] is None
            assert data['XFEL.H/I/J/K'] is not None
            assert data['XFEL.H/I/J/L'] is None

        with self.subTest("Receive data with invalid macropulse ID"):
            dataset["XFEL.H/I/J/K"] = ddgen.image(
                m._channels["XFEL.H/I/J/K"].value_schema(), pid=-1
            )
            data = m.query()
            assert data['XFEL.H/I/J/K'] is not None

    @patch("liso.experiment.doocs_interface.pydoocs_read")
    def testCorrelatedReadOnce(self, mocked_read):  # pylint: disable=too-many-statements
        logger.setLevel("WARNING")

        m = self._machine
        dataset = self._dataset
        mocked_read.side_effect = lambda x: _side_effect_read(dataset, x)

        with self.subTest("Normal"):
            # One channel has a different initial macropulse ID
            matched_pid = _INITIAL_PID + 3
            dataset["XFEL.A/B/C/D"] = ddgen.scalar(
                10., m._channels["XFEL.A/B/C/D"].value_schema(), pid=matched_pid)

            pid, data = m.read(1)[0]
            assert pid == matched_pid
            assert len(data) == 4
            assert data['XFEL.A/B/C/D']['data'] == 10.
            assert data['XFEL.A/B/C/D']['macropulse'] == matched_pid
            assert data['XFEL.A/B/C/E']['data'] == 100.
            assert data['XFEL.A/B/C/E']['macropulse'] == matched_pid
            np.testing.assert_array_equal(np.ones((4, 4)), data['XFEL.H/I/J/K']['data'])
            assert data['XFEL.H/I/J/K']['macropulse'] == 0
            np.testing.assert_array_equal(np.ones((5, 6)), data['XFEL.H/I/J/L']['data'])
            assert data['XFEL.H/I/J/L']['macropulse'] == matched_pid

        with self.subTest("Receive data with invalid macropulse ID"):
            dataset["XFEL.A/B/C/D"] = ddgen.scalar(
                    10., m._channels["XFEL.A/B/C/D"].value_schema(), pid=0)
            with self.assertLogs(level="WARNING") as cm:
                data = m.read(1)
            assert "macropulse == 0" in cm.output[0]
            assert len(data) == 0

            dataset["XFEL.A/B/C/D"] = ddgen.scalar(
                    10., m._channels["XFEL.A/B/C/D"].value_schema(), pid=-1)
            with self.assertLogs(level="WARNING") as cm:
                data = m.read(1)
            assert "macropulse == -1" in cm.output[0]
            assert len(data) == 0

            dataset["XFEL.A/B/C/D"] = ddgen.scalar(
                    10., m._channels["XFEL.A/B/C/D"].value_schema(), pid=m._corr._last_correlated)

        with self.subTest("Raise when reading non-event channel"):
            assert "XFEL.H/I/J/K" in m._non_event

            def _side_effect_read2(dataset, address):
                if address == "XFEL.H/I/J/K":
                    raise np.random.choice([PyDoocsException, DoocsException])
                return dataset[address]
            mocked_read.side_effect = lambda x: _side_effect_read2(dataset, x)
            with self.assertLogs(level="ERROR") as cm:
                data = m.read(1)
            assert "Failed to read data from XFEL.H/I/J/K" in cm.output[0]
            assert len(data) == 0

    @patch("liso.experiment.doocs_interface.pydoocs_read")
    def testCorrelatedReadOnceWithOldPulseId(self, patched_read):
        m = self._machine
        dataset = self._dataset
        patched_read.side_effect = lambda x: _side_effect_read(dataset, x)

        with self.subTest("Event channel has old macropulse ID"):
            last_correlated_gt = _INITIAL_PID
            m._corr._last_correlated = _INITIAL_PID
            dataset["XFEL.A/B/C/D"]['macropulse'] = last_correlated_gt - 500
            data = m.read(1)
            assert len(data) == 0
            self.assertEqual(last_correlated_gt, m._corr._last_correlated)

        with self.subTest("Non-event channel has normal macropulse ID"):
            for address in dataset:
                dataset[address]['macropulse'] = last_correlated_gt
            data = m.read(1)
            assert len(data) == 1
            assert data[0][0] == last_correlated_gt + 1
            self.assertLess(last_correlated_gt, m._corr._last_correlated)

    @patch("liso.experiment.doocs_interface.pydoocs_read")
    def testCorrelatedReadMultiple(self, patched_read):
        m = self._machine
        dataset = self._dataset
        patched_read.side_effect = lambda x: _side_effect_read(dataset, x)

        items = m.read(4)
        assert len(items) == 4
        assert [pid for pid, _ in items] == [1000, 1001, 1002, 1003]

    @patch("time.sleep", side_effect=KeyboardInterrupt)
    @patch("liso.experiment.doocs_interface.pydoocs_read")
    def testAcquire(self, patched_read, _):
        m = self._machine
        patched_read.side_effect = lambda x: _side_effect_read(self._dataset, x)

        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(TypeError, "type 'any' not understood"):
                m.acquire(tmp_dir)

            del m._channels["XFEL.A/B/C/E"]
            with self.assertRaisesRegex(TypeError, "Validation error"):
                m.add_control_channel("XFEL.A/B/C/E", dc.INT)
                m.acquire(tmp_dir)

            del m._channels["XFEL.A/B/C/E"]
            m.add_control_channel("XFEL.A/B/C/E", dc.FLOAT)
            logger.setLevel("INFO")
            with self.assertLogs(level="INFO") as cm:
                m.acquire(tmp_dir)
            assert "Saved 1 pulse" in cm.output[-1]
            run = open_run(Path(tmp_dir).joinpath('r0003'))
            run.info()

    @patch("time.sleep", side_effect=KeyboardInterrupt)
    @patch("liso.experiment.doocs_interface.pydoocs_read")
    def testMonitor(self, mocked_pydoocs_read, _):
        m = self._machine
        mocked_pydoocs_read.side_effect = lambda x: _side_effect_read(self._dataset, x)

        with patch.object(m, "query") as mocked_query:
            self._machine.monitor()
            mocked_query.assert_called_once()

        with patch.object(m, "read") as mocked_read:
            self._machine.monitor(correlate=True)
            mocked_read.assert_called_once()

        self._machine.monitor()
        self._machine.monitor(verbose=False)

        self._machine.monitor(correlate=True)
        self._machine.monitor(verbose=False)
