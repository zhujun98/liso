import unittest
from unittest.mock import patch

import numpy as np

from liso import EuXFELInterface
from liso import doocs_channels as dc


class TestDoocsMachine(unittest.TestCase):
    def setUp(self):
        m = EuXFELInterface(delay=0.01)
        m.add_control_channel(dc.FLOAT32, "A/B/C/D")
        m.add_control_channel(dc.FLOAT, "A/B/C/E")
        m.add_instrument_channel(dc.IMAGE, "H/I/J/K", shape=(4, 4), dtype="uint16")
        m.add_instrument_channel(dc.IMAGE, "H/I/J/L", shape=(5, 6), dtype="float32")
        self._machine = m

    def testChannelManipulation(self):
        m = self._machine

        self.assertListEqual(["A/B/C/D", 'A/B/C/E'], m.controls)
        self.assertListEqual(["H/I/J/K", 'H/I/J/L'], m.instruments)
        self.assertListEqual(m.controls + m.instruments, m.channels)

        with self.subTest("Add an existing channel"):
            with self.assertRaises(ValueError):
                m.add_control_channel(dc.IMAGE, "A/B/C/D", shape=(2, 2), dtype="uint16")
            with self.assertRaises(ValueError):
                m.add_control_channel(dc.IMAGE, "H/I/J/K", shape=(2, 2), dtype="uint16")
            with self.assertRaises(ValueError):
                m.add_instrument_channel(dc.FLOAT, "A/B/C/D")
            with self.assertRaises(ValueError):
                m.add_instrument_channel(dc.FLOAT, "H/I/J/K")

        with self.subTest("Test schema"):
            m = self._machine
            control_schema, instrument_schema = m.schema
            self.assertDictEqual(
                {'A/B/C/D': {'default': 0.0, 'type': '<f4',
                             'maximum': np.finfo(np.float32).max,
                             'minimum': np.finfo(np.float32).min},
                 'A/B/C/E': {'default': 0.0, 'type': '<f8'}},
                control_schema
            )
            self.assertDictEqual(
                {'H/I/J/K': {'dtype': '<u2', 'shape': (4, 4), 'type': 'NDArray'},
                 'H/I/J/L': {'dtype': '<f4', 'shape': (5, 6), 'type': 'NDArray'}},
                instrument_schema
            )

    @patch("liso.experiment.machine.pydoocs_write")
    @patch("liso.experiment.machine.pydoocs_read")
    def testRun(self, patched_read, patched_write):
        self._machine.run()
        self.assertEqual(len(self._machine.channels), patched_read.call_count)

        self._machine.run(mapping={'A/B/C/D': 5})
        patched_write.assert_called_once_with('A/B/C/D', 5)
