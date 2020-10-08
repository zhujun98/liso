import unittest

from liso.experiments import Channel


class TestDoocs(unittest.TestCase):
    def testChannel(self):
        ch1 = Channel.from_address(
            "XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L2/SUMVOLTAGE.CHIRP.SP.1")
        self.assertEqual("XFEL.RF", ch1.facility)
        self.assertEqual("LLRF.SUMVOLTAGE_CTRL", ch1.device)
        self.assertEqual("L2", ch1.location)
        self.assertEqual("SUMVOLTAGE.CHIRP.SP.1", ch1.property)

        with self.assertRaises(ValueError):
            Channel.from_address("XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L2/")
        with self.assertRaises(ValueError):
            Channel.from_address("XFEL.RF/LLRF.SUMVOLTAGE_CTRL//SUMVOLTAGE.CHIRP.SP.1")
