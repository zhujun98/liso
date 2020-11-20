import unittest
from unittest.mock import patch

import numpy as np

from liso import EuXFELInterface
from liso import doocs_channels as dc
from liso.exceptions import LisoRuntimeError
from liso.experiment import machine
from liso.experiment.machine import _DoocsReader, _DoocsWriter
from liso.logging import logger
logger.setLevel("ERROR")

from . import DoocsDataGenerator as ddgen


_PID0 = 1000


def _side_effect_read(dataset, address, error=0):
    data = dataset[address]
    if data['macropulse'] >= _PID0:
        if np.random.rand() > 0.5:
            data['macropulse'] += 1
            # do not mutate
            data['data'] = data['data'] + error
    return data


class TestDoocsMachine(unittest.TestCase):
    def run(self, result=None):
        DELAY_NO_EVENT = _DoocsReader._DELAY_NO_EVENT
        DELAY_STALE = _DoocsReader._DELAY_STALE
        DELAY_EXCEPTION = _DoocsReader._DELAY_EXCEPTION

        WRITE_DELAY_EXCEPTION = _DoocsWriter._DELAY_EXCEPTION

        try:
            _DoocsReader._DELAY_NO_EVENT = 1e-3
            _DoocsReader._DELAY_STALE = 1e-4
            _DoocsReader._DELAY_EXCEPTION = 5e-4

            _DoocsWriter._DELAY_EXCEPTION = 5e-4
            super().run(result)
        finally:
            _DoocsReader._DELAY_NO_EVENT = DELAY_NO_EVENT
            _DoocsReader._DELAY_STALE = DELAY_STALE
            _DoocsReader._DELAY_EXCEPTION = DELAY_EXCEPTION

            _DoocsWriter._DELAY_EXCEPTION = WRITE_DELAY_EXCEPTION

    def setUp(self):
        m = EuXFELInterface()
        m.add_control_channel(dc.FLOAT, "XFEL.A/B/C/D")
        m.add_control_channel(dc.DOUBLE, "XFEL.A/B/C/E")
        m.add_diagnostic_channel(dc.IMAGE, "XFEL.H/I/J/K",
                                 shape=(4, 4), dtype="uint16", no_event=True)
        m.add_diagnostic_channel(dc.IMAGE, "XFEL.H/I/J/L",
                                 shape=(5, 6), dtype="float32")
        self._machine = m

        self._dataset = {
            "XFEL.A/B/C/D": ddgen.scalar(
                10., m._controls["XFEL.A/B/C/D"].value_schema(), pid=_PID0),
            "XFEL.A/B/C/E": ddgen.scalar(
                100., m._controls["XFEL.A/B/C/E"].value_schema(), pid=_PID0),
            # default value is 1
            "XFEL.H/I/J/K": ddgen.image(
                m._diagnostics["XFEL.H/I/J/K"].value_schema(), pid=100),
            "XFEL.H/I/J/L": ddgen.image(
                m._diagnostics["XFEL.H/I/J/L"].value_schema(), pid=_PID0)
        }

    def testChannelManipulation(self):
        m = self._machine

        self.assertListEqual(["XFEL.A/B/C/D", 'XFEL.A/B/C/E'], m.controls)
        self.assertListEqual(["XFEL.H/I/J/K", 'XFEL.H/I/J/L'], m.diagnostics)
        self.assertListEqual(m.controls + m.diagnostics, m.channels)
        self.assertSetEqual(set(m.controls + m.diagnostics), m._reader._channels)
        self.assertSetEqual({"XFEL.H/I/J/K"}, m._reader._no_event)

        with self.subTest("Add an existing channel"):
            with self.assertRaises(ValueError):
                m.add_control_channel(dc.IMAGE, "XFEL.A/B/C/D",
                                      shape=(2, 2), dtype="uint16")
            with self.assertRaises(ValueError):
                m.add_control_channel(dc.IMAGE, "XFEL.H/I/J/K",
                                      shape=(2, 2), dtype="uint16")
            with self.assertRaises(ValueError):
                m.add_diagnostic_channel(dc.FLOAT, "XFEL.A/B/C/D")
            with self.assertRaises(ValueError):
                m.add_diagnostic_channel(dc.FLOAT, "XFEL.H/I/J/K")

        with self.subTest("Invalid address"):
            with self.assertRaisesRegex(ValueError, "must start with XFEL"):
                m.add_control_channel(dc.FLOAT, "A/B/C/D")

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

    @patch("liso.experiment.machine.pydoocs_write")
    @patch("liso.experiment.machine.pydoocs_read")
    def testRun(self, patched_read, patched_write):
        dataset = self._dataset
        patched_read.side_effect = lambda x: _side_effect_read(dataset, x)

        self._machine.write_and_read(timeout=0.01)
        self.assertGreaterEqual(patched_read.call_count, len(self._machine.channels))

        with self.subTest("Test readout"):
            # 'readout' is empty or None
            self._machine.write_and_read(mapping={'XFEL.A/B/C/D': {
                'value': 5., 'readout': None, 'tol': 1e-6}})
            patched_write.assert_called_once_with('XFEL.A/B/C/D', 5)
            patched_write.reset_mock()
            self._machine.write_and_read(mapping={'XFEL.A/B/C/D': {
                'value': 5, 'tol': 1e-6}})
            patched_write.assert_called_once_with('XFEL.A/B/C/D', 5)
            patched_write.reset_mock()

            # 'readout' is a registered channel
            self._machine.write_and_read(mapping={'XFEL.A/B/C/D': {
                'value': 10, 'readout': 'XFEL.A/B/C/D', 'tol': 1e-6}})
            patched_write.assert_called_once_with('XFEL.A/B/C/D', 10)

            # 'readout' is not registered
            with self.assertRaisesRegex(ValueError, "not been registered"):
                self._machine.write_and_read(mapping={'XFEL.A/B/C/D': {
                    'value': 10, 'readout': 'XFEL.A/B/C/F', 'tol': 1e-6}})

        with self.subTest("Test validation"):
            orig_v = dataset["XFEL.A/B/C/D"]['data']
            with self.assertRaisesRegex(LisoRuntimeError, 'ValidationError'):
                dataset["XFEL.A/B/C/D"]['data'] = 1
                self._machine.write_and_read()
            dataset["XFEL.A/B/C/D"]['data'] = orig_v

            orig_v = dataset["XFEL.H/I/J/K"]['data']
            with self.assertRaisesRegex(LisoRuntimeError, 'ValidationError'):
                dataset["XFEL.H/I/J/K"]['data'] = np.ones((2, 2))
                self._machine.write_and_read()
            dataset["XFEL.H/I/J/K"]['data'] = orig_v

        with self.subTest("Test raise when reading"):
            # raise happens to an event-based channel
            def _side_effect_read2(dataset, address):
                if address != "XFEL.H/I/J/K":
                    raise np.random.choice([machine.PyDoocsException,
                                            machine.DoocsException])
                return dataset[address]
            patched_read.side_effect = lambda x: _side_effect_read2(dataset, x)
            with self.assertRaisesRegex(LisoRuntimeError, 'Unable to match all data'):
                self._machine.write_and_read(timeout=0.1)

            # raise happens to a no-event channel
            def _side_effect_read3(dataset, address):
                if address == "XFEL.H/I/J/K":
                    raise np.random.choice([machine.PyDoocsException,
                                            machine.DoocsException])
                data = dataset[address]
                if data['macropulse'] >= _PID0:
                    if np.random.rand() > 0.5:
                        data['macropulse'] += 1

                return dataset[address]
            patched_read.side_effect = lambda x: _side_effect_read3(dataset, x)
            with self.assertRaisesRegex(LisoRuntimeError, 'XFEL.H/I/J/K'):
                self._machine.write_and_read(timeout=0.1)

        with self.subTest("Test raise when writing"):
            with self.assertRaisesRegex(LisoRuntimeError,
                                        "Failed to write new values to all channels"):
                def _side_effect_write(address, v):
                    if address == 'XFEL.A/B/C/E':
                        raise np.random.choice([machine.PyDoocsException,
                                                machine.DoocsException])
                patched_write.side_effect = _side_effect_write
                self._machine.write_and_read(mapping={
                    'XFEL.A/B/C/D': {
                        'value': 10, 'readout': 'XFEL.A/B/C/D', 'tol': 1e-6
                    },
                    'XFEL.A/B/C/E': {
                        'value': 100, 'readout': 'XFEL.A/B/C/D', 'tol': 1e-5
                    },
                })

    @patch("liso.experiment.machine.pydoocs_write")
    @patch("liso.experiment.machine.pydoocs_read")
    def testCorrelation(self, patched_read, patched_write):
        dataset = self._dataset
        patched_read.side_effect = lambda x: _side_effect_read(dataset, x, error=1)

        dataset["XFEL.A/B/C/D"] = ddgen.scalar(
                1., self._machine._controls["XFEL.A/B/C/D"].value_schema(), pid=1001)

        # add an implicit non-event based data
        self._machine.add_control_channel(dc.FLOAT, "XFEL.A/B/C/F")
        dataset["XFEL.A/B/C/F"] = ddgen.scalar(
                1., self._machine._controls["XFEL.A/B/C/D"].value_schema(), pid=0)

        pid, control_data, diagnostic_data = self._machine.write_and_read()
        self.assertDictEqual({'XFEL.A/B/C/D': pid - _PID0,
                              'XFEL.A/B/C/E': pid - _PID0 + 100,
                              'XFEL.A/B/C/F': 1.0},
                             control_data)
        np.testing.assert_array_equal(np.ones((4, 4)), diagnostic_data['XFEL.H/I/J/K'])
        np.testing.assert_array_equal((pid - _PID0 + 1) * np.ones((5, 6)),
                                      diagnostic_data['XFEL.H/I/J/L'])

        with self.subTest("Test receiving data with invalid macropulse ID"):
            dataset["XFEL.A/B/C/D"] = ddgen.scalar(
                    1., self._machine._controls["XFEL.A/B/C/D"].value_schema(), pid=-1)
            with self.assertRaisesRegex(LisoRuntimeError, 'Unable to match'):
                self._machine.write_and_read(timeout=0.1)

    @patch("liso.experiment.machine.pydoocs_write")
    @patch("liso.experiment.machine.pydoocs_read")
    def testCorrelationWithOldPulseId(self, patched_read, patched_write):
        dataset = self._dataset
        patched_read.side_effect = lambda x: _side_effect_read(dataset, x)

        reader = self._machine._reader
        last_correlated_gt = 1000
        reader._last_correlated = 1000
        for address in dataset:
            dataset[address]['macropulse'] = last_correlated_gt - 500
        with self.assertRaisesRegex(LisoRuntimeError, 'Unable to match'):
            self._machine.write_and_read(timeout=0.02)
        self.assertEqual(last_correlated_gt, reader._last_correlated)

        for address in dataset:
            dataset[address]['macropulse'] = last_correlated_gt
        self._machine.write_and_read(timeout=0.02)
        self.assertLess(last_correlated_gt, reader._last_correlated)

    @patch("liso.experiment.machine.pydoocs_write")
    @patch("liso.experiment.machine.pydoocs_read")
    def testCheckReadoutAfterWriting(self, patched_read, patched_write):
        dataset = self._dataset
        patched_read.side_effect = lambda x: _side_effect_read(dataset, x, error=1)

        self._machine.write_and_read(mapping={
            'XFEL.A/B/C/D': {
                'value': 19.91, 'readout': 'XFEL.A/B/C/D', 'tol': 0.1
            },
        })

        with self.assertRaisesRegex(LisoRuntimeError, 'Unable to match'):
            # The value below should be much larger than the value above.
            # The actual value should get to 30, but 30 - 29.9 = 0.1
            self._machine.write_and_read(mapping={
                'XFEL.A/B/C/D': {
                    'value': 29.9, 'readout': 'XFEL.A/B/C/D', 'tol': 0.1
                },
            }, timeout=0.02)

    @patch("liso.experiment.machine.pydoocs_read")
    def testTakeSnapshot(self, patched_read):
        dataset = self._dataset
        patched_read.side_effect = lambda x: _side_effect_read(dataset, x)

        snapshot = self._machine.take_snapshot(self._machine.channels)
        self.assertSetEqual(
            {"XFEL.A/B/C/D", 'XFEL.A/B/C/E', "XFEL.H/I/J/K", 'XFEL.H/I/J/L'},
            set(snapshot)
        )
