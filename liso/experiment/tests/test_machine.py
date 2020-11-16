import unittest
from unittest.mock import patch

import numpy as np

from liso import EuXFELInterface
from liso import doocs_channels as dc
from liso.exceptions import LisoRuntimeError
from liso.experiment import machine
from liso.logging import logger
logger.setLevel("CRITICAL")

from . import DoocsDataGenerator as ddgen


def _side_effect_read(dataset, address, error=0):
    data = dataset[address]
    if data['macropulse'] > 0:
        data['macropulse'] += 1
    data['data'] += error
    return data


class TestDoocsMachine(unittest.TestCase):
    def setUp(self):
        m = EuXFELInterface(delay=0.01)
        m.add_control_channel(dc.FLOAT, "XFEL.A/B/C/D")
        m.add_control_channel(dc.DOUBLE, "XFEL.A/B/C/E")
        m.add_diagnostic_channel(dc.IMAGE, "XFEL.H/I/J/K", shape=(4, 4), dtype="uint16", no_event=True)
        m.add_diagnostic_channel(dc.IMAGE, "XFEL.H/I/J/L", shape=(5, 6), dtype="float32")
        self._machine = m

        self._dataset = {
            "XFEL.A/B/C/D": ddgen.scalar(
                10., m._controls["XFEL.A/B/C/D"].value_schema(), pid=1000),
            "XFEL.A/B/C/E": ddgen.scalar(
                100., m._controls["XFEL.A/B/C/E"].value_schema(), pid=1000),
            "XFEL.H/I/J/K": ddgen.image(
                m._diagnostics["XFEL.H/I/J/K"].value_schema(), pid=1),
            "XFEL.H/I/J/L": ddgen.image(
                m._diagnostics["XFEL.H/I/J/L"].value_schema(), pid=1000)
        }

    def testChannelManipulation(self):
        m = self._machine

        self.assertListEqual(["XFEL.A/B/C/D", 'XFEL.A/B/C/E'], m.controls)
        self.assertListEqual(["XFEL.H/I/J/K", 'XFEL.H/I/J/L'], m.diagnostics)
        self.assertListEqual(m.controls + m.diagnostics, m.channels)

        with self.subTest("Add an existing channel"):
            with self.assertRaises(ValueError):
                m.add_control_channel(dc.IMAGE, "XFEL.A/B/C/D", shape=(2, 2), dtype="uint16")
            with self.assertRaises(ValueError):
                m.add_control_channel(dc.IMAGE, "XFEL.H/I/J/K", shape=(2, 2), dtype="uint16")
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

        self._machine.run()
        self.assertEqual(len(self._machine.channels), patched_read.call_count)

        # 'readout' is empty or None
        self._machine.run(mapping={'XFEL.A/B/C/D': {
            'value': 5., 'readout': None, 'tol': 1e-6}})
        patched_write.assert_called_once_with('XFEL.A/B/C/D', 5)
        patched_write.reset_mock()
        self._machine.run(mapping={'XFEL.A/B/C/D': {
            'value': 5, 'tol': 1e-6}})
        patched_write.assert_called_once_with('XFEL.A/B/C/D', 5)
        patched_write.reset_mock()

        # 'readout' is a registered channel
        self._machine.run(mapping={'XFEL.A/B/C/D': {
            'value': 10, 'readout': 'XFEL.A/B/C/D', 'tol': 1e-6}})
        patched_write.assert_called_once_with('XFEL.A/B/C/D', 10)

        # 'readout' is not registered
        with self.assertRaisesRegex(ValueError, "not been registered"):
            self._machine.run(mapping={'XFEL.A/B/C/D': {
                'value': 10, 'readout': 'XFEL.A/B/C/F', 'tol': 1e-6}})

        with self.subTest("Test validation"):
            orig_v = dataset["XFEL.A/B/C/D"]['data']
            with self.assertRaisesRegex(LisoRuntimeError, 'ValidationError'):
                dataset["XFEL.A/B/C/D"]['data'] = 1
                self._machine.run()
            dataset["XFEL.A/B/C/D"]['data'] = orig_v

            orig_v = dataset["XFEL.H/I/J/K"]['data']
            with self.assertRaisesRegex(LisoRuntimeError, 'ValidationError'):
                dataset["XFEL.H/I/J/K"]['data'] = np.ones((2, 2))
                self._machine.run()
            dataset["XFEL.H/I/J/K"]['data'] = orig_v

        with self.subTest("Test raise when reading"):
            def _side_effect_read2(dataset, address):
                if np.random.rand() > 0.7:
                    raise np.random.choice([machine.PyDoocsException,
                                            machine.DoocsException])
                return dataset[address]
            patched_read.side_effect = lambda x: _side_effect_read2(dataset, x)
            with self.assertRaisesRegex(LisoRuntimeError, 'Unable to match'):
                self._machine.run()

        with self.subTest("Test raise when writing"):
            with self.assertRaisesRegex(LisoRuntimeError,
                                        "Failed to write new values to all channels"):
                def _side_effect_write(address, v):
                    if address == 'XFEL.A/B/C/E':
                        raise np.random.choice([machine.PyDoocsException,
                                                machine.DoocsException])
                patched_write.side_effect = _side_effect_write
                self._machine.run(mapping={
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

        pid, control_data, diagnostic_data = self._machine.run(max_attempts=2)
        self.assertEqual(1002, pid)
        self.assertDictEqual({'XFEL.A/B/C/D': 2.0, 'XFEL.A/B/C/E': 102.0, 'XFEL.A/B/C/F': 3.0},
                             control_data)
        np.testing.assert_array_equal(3 * np.ones((4, 4)), diagnostic_data['XFEL.H/I/J/K'])
        np.testing.assert_array_equal(3 * np.ones((5, 6)), diagnostic_data['XFEL.H/I/J/L'])

        with self.assertRaisesRegex(LisoRuntimeError, 'Unable to match'):
            self._machine.run(max_attempts=1)

        with self.subTest("Test checking written data"):
            self._machine.run(mapping={
                'XFEL.A/B/C/D': {
                    'value': 4.9, 'readout': 'XFEL.A/B/C/D', 'tol': 0.1
                },
            }, max_attempts=2)
            with self.assertRaisesRegex(LisoRuntimeError, 'Unable to match'):
                self._machine.run(mapping={
                    'XFEL.A/B/C/D': {
                        'value': 6.9, 'readout': 'XFEL.A/B/C/D', 'tol': 0.09
                    },
                }, max_attempts=10)

        with self.subTest("Test receiving data with invalid macropulse ID"):
            dataset["XFEL.A/B/C/D"] = ddgen.scalar(
                    1., self._machine._controls["XFEL.A/B/C/D"].value_schema(), pid=-1)
            with self.assertRaisesRegex(LisoRuntimeError, 'Unable to match'):
                self._machine.run(max_attempts=10)

        with self.subTest("Test receiving non-event based data (ID = 0)"):
            dataset["XFEL.A/B/C/D"] = ddgen.scalar(
                    1., self._machine._controls["XFEL.A/B/C/D"].value_schema(), pid=0)
            self._machine.run(max_attempts=1)

        with self.subTest("Test receiving data with pulse ID smaller than the correlated one"):
            last_correlated_gt = 1026
            self.assertEqual(last_correlated_gt, self._machine._last_correlated)
            for address in dataset:
                dataset[address]['macropulse'] = last_correlated_gt - 1
            with self.assertRaisesRegex(LisoRuntimeError, 'Unable to match'):
                self._machine.run(max_attempts=1)
            self.assertEqual(last_correlated_gt, self._machine._last_correlated)
            self._machine.run(max_attempts=2)
            self.assertEqual(last_correlated_gt + 1, self._machine._last_correlated)
