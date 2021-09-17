import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from liso import EuXFELInterface
from liso import doocs_channels as dc
from liso.exceptions import LisoRuntimeError
from liso.experiment import doocs_channels
from liso.experiment.doocs_interface import DoocsException, PyDoocsException
from liso.logging import logger

from . import DoocsDataGenerator as ddgen


_INITIAL_PID = 1000


def _side_effect_read(dataset, address):
    data = dataset[address].copy()
    if data['macropulse'] >= _INITIAL_PID:
        dataset[address]['macropulse'] += 1
    return data


class TestDoocsInterface(unittest.TestCase):
    def setUp(self):
        logger.setLevel("ERROR")

        cfg = {
            "timeout.correlation": 0.1,
            "interval.read.non_event_date": 0.02,
            "interval.read.retry": 0.01
        }
        m = EuXFELInterface(cfg)
        m.add_control_channel(dc.FLOAT, "XFEL.A/B/C/D", "XFEL.A/B/C/d")
        m.add_control_channel(dc.DOUBLE, "XFEL.A/B/C/E")
        m.add_diagnostic_channel(dc.IMAGE, "XFEL.H/I/J/K",
                                 shape=(4, 4), dtype="uint16", non_event=True)
        m.add_diagnostic_channel(dc.IMAGE, "XFEL.H/I/J/L",
                                 shape=(5, 6), dtype="float32")
        self._machine = m

        self._dataset = {
            "XFEL.A/B/C/D": ddgen.scalar(
                10., m._controls["XFEL.A/B/C/D"].value_schema(), pid=_INITIAL_PID),
            "XFEL.A/B/C/E": ddgen.scalar(
                100., m._controls["XFEL.A/B/C/E"].value_schema(), pid=_INITIAL_PID),
            "XFEL.H/I/J/K": ddgen.image(
                m._diagnostics["XFEL.H/I/J/K"].value_schema(), pid=0),  # non-event
            "XFEL.H/I/J/L": ddgen.image(
                m._diagnostics["XFEL.H/I/J/L"].value_schema(), pid=_INITIAL_PID)
        }

    def testChannelManipulation(self):
        m = self._machine

        self.assertListEqual(["XFEL.A/B/C/D", 'XFEL.A/B/C/E'], m.controls)
        self.assertListEqual(["XFEL.A/B/C/d", 'XFEL.A/B/C/E'], list(m._controls_write.values()))
        self.assertListEqual(["XFEL.H/I/J/K", 'XFEL.H/I/J/L'], m.diagnostics)
        self.assertListEqual(m.controls + m.diagnostics, m.channels)
        self.assertSetEqual({"XFEL.H/I/J/K"}, m._non_event)

        with self.subTest("Add an existing channel"):
            with self.assertRaisesRegex(ValueError, "control"):
                m.add_control_channel(dc.IMAGE, "XFEL.A/B/C/D",
                                      shape=(2, 2), dtype="uint16")
            with self.assertRaisesRegex(ValueError, "diagnostics"):
                m.add_control_channel(dc.IMAGE, "XFEL.H/I/J/K",
                                      shape=(2, 2), dtype="uint16")
            with self.assertRaisesRegex(ValueError, "control"):
                m.add_diagnostic_channel(dc.FLOAT, "XFEL.A/B/C/D")
            with self.assertRaisesRegex(ValueError, "diagnostics"):
                m.add_diagnostic_channel(dc.FLOAT, "XFEL.H/I/J/K")

        with self.subTest("Test schema"):
            m = self._machine
            control_schema, diagnostic_schema = m.schema
            self.assertDictEqual(
                {'XFEL.A/B/C/D': {'default': 0.0, 'type': '<f4',
                                  'maximum': np.finfo(np.float32).max,
                                  'minimum': np.finfo(np.float32).min},
                 'XFEL.A/B/C/E': {'default': 0.0, 'type': '<f8'}},
                control_schema
            )
            self.assertDictEqual(
                {'XFEL.H/I/J/K': {'dtype': '<u2', 'shape': (4, 4), 'type': 'NDArray'},
                 'XFEL.H/I/J/L': {'dtype': '<f4', 'shape': (5, 6), 'type': 'NDArray'}},
                diagnostic_schema
            )

    @patch("liso.experiment.doocs_interface.pydoocs_write")
    def testWrite(self, patched_write):
        m = self._machine

        with self.assertRaisesRegex(KeyError, "not found in the control channels"):
            m.write(mapping={'XFEL.A/B/C/C': 1.})

        with self.assertRaisesRegex(LisoRuntimeError, "Failed to update 1/2 channels"):
            with self.assertLogs(level="ERROR") as cm:
                def _side_effect_write1(address, v):
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
                def _side_effect_write2(address, v):
                    if address == 'XFEL.A/B/C/d':
                        raise np.random.choice([ValueError, RuntimeError])
                patched_write.side_effect = _side_effect_write2
                m.write(mapping={
                    'XFEL.A/B/C/D': 1.,
                    'XFEL.A/B/C/E': 10.,
                })
        assert "Unexpected exception" in cm.output[0]

    @patch("liso.experiment.doocs_interface.pydoocs_read")
    def testUncorrelatedRead(self, patched_read):
        m = self._machine
        dataset = self._dataset
        patched_read.side_effect = lambda x: _side_effect_read(dataset, x)

        with self.subTest("Test normal"):
            pid, control_data, diagnostic_data = self._machine.read(correlate=False)
            assert patched_read.call_count == 4
            assert pid is None
            assert len(control_data) == 2
            assert len(diagnostic_data) == 2
            assert control_data['XFEL.A/B/C/D']['data'] == 10
            assert control_data['XFEL.A/B/C/E']['macropulse'] in [1000, 1001]
            assert diagnostic_data['XFEL.H/I/J/K']['macropulse'] == 0
            assert diagnostic_data['XFEL.H/I/J/L']['type']== 'IMAGE'

        with self.subTest("Test raise when reading"):
            # raise happens to an event-based channel
            def _side_effect_read2(dataset, address):
                if address != "XFEL.H/I/J/K":
                    raise np.random.choice([PyDoocsException, DoocsException])
                return dataset[address]
            patched_read.side_effect = lambda x: _side_effect_read2(dataset, x)
            pid, control_data, diagnostic_data = m.read(correlate=False)
            assert len(control_data) == 2
            assert len(diagnostic_data) == 2
            assert control_data['XFEL.A/B/C/D'] is None
            assert control_data['XFEL.A/B/C/E'] is None
            assert diagnostic_data['XFEL.H/I/J/K'] is not None
            assert diagnostic_data['XFEL.H/I/J/L'] is None

        with self.subTest("Test receiving data with invalid macropulse ID"):
            dataset["XFEL.H/I/J/K"] = ddgen.image(
                m._diagnostics["XFEL.H/I/J/K"].value_schema(), pid=-1
            )
            m.read(correlate=False)
            assert diagnostic_data['XFEL.H/I/J/K'] is not None

    @patch("liso.experiment.doocs_interface.pydoocs_read")
    def testValidationRead(self, patched_read):
        m = self._machine
        dataset = self._dataset
        patched_read.side_effect = lambda x: _side_effect_read(dataset, x)

        orig_v = dataset["XFEL.A/B/C/D"]['data']
        with self.assertRaisesRegex(LisoRuntimeError, 'ValidationError'):
            dataset["XFEL.A/B/C/D"]['data'] = 1
            m.read()
        dataset["XFEL.A/B/C/D"]['data'] = orig_v

        orig_v = dataset["XFEL.A/B/C/E"]['data']
        with self.assertRaisesRegex(LisoRuntimeError, 'ValidationError'):
            dataset["XFEL.A/B/C/E"]['data'] = True
            m.read(correlate=False)
        dataset["XFEL.A/B/C/E"]['data'] = orig_v

        orig_v = dataset["XFEL.H/I/J/K"]['data']
        with self.assertRaisesRegex(LisoRuntimeError, 'ValidationError'):
            dataset["XFEL.H/I/J/K"]['data'] = np.ones((2, 2))
            m.read()
        # turn validation off
        m.read(validate=False)
        dataset["XFEL.H/I/J/K"]['data'] = orig_v

    @patch("liso.experiment.doocs_interface.pydoocs_read")
    def testCorrelatedRead(self, patched_read):
        logger.setLevel("WARNING")

        m = self._machine
        if sys.platform == "darwin":
            m._timeout_correlating = 0.2
        dataset = self._dataset
        patched_read.side_effect = lambda x: _side_effect_read(dataset, x)

        with self.subTest("Test normal"):
            # One channel has a different initial macropulse ID
            matched_pid = _INITIAL_PID + 3
            dataset["XFEL.A/B/C/D"] = ddgen.scalar(
                10., m._controls["XFEL.A/B/C/D"].value_schema(), pid=matched_pid)

            pid, control_data, diagnostic_data = m.read()
            assert pid == matched_pid
            assert len(control_data) == 2
            assert len(diagnostic_data) == 2
            assert control_data['XFEL.A/B/C/D']['data'] == 10.
            assert control_data['XFEL.A/B/C/D']['macropulse'] == matched_pid
            assert control_data['XFEL.A/B/C/E']['data'] == 100.
            assert control_data['XFEL.A/B/C/E']['macropulse'] == matched_pid
            np.testing.assert_array_equal(np.ones((4, 4)), diagnostic_data['XFEL.H/I/J/K']['data'])
            assert diagnostic_data['XFEL.H/I/J/K']['macropulse'] == 0
            np.testing.assert_array_equal(np.ones((5, 6)), diagnostic_data['XFEL.H/I/J/L']['data'])
            assert diagnostic_data['XFEL.H/I/J/L']['macropulse'] == matched_pid

        with self.subTest("Test receiving data with invalid macropulse ID"):
            dataset["XFEL.A/B/C/D"] = ddgen.scalar(
                    10., m._controls["XFEL.A/B/C/D"].value_schema(), pid=0)
            with self.assertRaisesRegex(LisoRuntimeError, 'Failed to correlate'):
                with self.assertLogs("WARNING") as cm:
                    m.read()
                assert "macropuse == 0" in cm.output[0]

            dataset["XFEL.A/B/C/D"] = ddgen.scalar(
                    10., m._controls["XFEL.A/B/C/D"].value_schema(), pid=-1)
            with self.assertRaisesRegex(LisoRuntimeError, 'Failed to correlate'):
                with self.assertLogs("WARNING") as cm:
                    m.read()
                assert "macropuse == -1" in cm.output[0]

        # # raise happens to a no-event channel
        # def _side_effect_read3(dataset, address):
        #     if address == "XFEL.H/I/J/K":
        #         raise np.random.choice([PyDoocsException, DoocsException])
        #     data = dataset[address]
        #     if data['macropulse'] >= _INITIAL_PID and np.random.rand() > 0.5:
        #             data['macropulse'] += 1
        #     return dataset[address]
        #
        # patched_read.side_effect = lambda x: _side_effect_read3(dataset, x)
        # self._machine.read()

        #
        # with self.subTest("Test monitor"):
        #     pass

    @patch("liso.experiment.doocs_interface.pydoocs_read")
    def testCorrelationWithOldPulseId(self, patched_read):
        m = self._machine
        dataset = self._dataset
        patched_read.side_effect = lambda x: _side_effect_read(dataset, x)

        last_correlated_gt = 1000
        m._last_correlated = 1000
        for address in dataset:
            dataset[address]['macropulse'] = last_correlated_gt - 500
        with self.assertRaisesRegex(LisoRuntimeError, 'Failed to correlate'):
            m.read()
        self.assertEqual(last_correlated_gt, m._last_correlated)

        for address in dataset:
            dataset[address]['macropulse'] = last_correlated_gt
        m.read()
        self.assertLess(last_correlated_gt, m._last_correlated)

    @patch("time.sleep", side_effect=KeyboardInterrupt)
    def testMonitor(self, patched_sleep):

        mocked_read = MagicMock(return_value=(None, dict(), dict()))
        self._machine.read = mocked_read

        self._machine.monitor()
        self.assertDictEqual({'correlate': False, 'validate': True},
                             mocked_read.call_args_list[0][1])
        patched_sleep.assert_called_with(1.0)
        patched_sleep.reset_mock()
