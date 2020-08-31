import unittest

import numpy as np

from liso.scan import LinacScan
from liso.simulation import Linac


class TestLinacscan(unittest.TestCase):
    def setUp(self):
        self._linac = Linac()  # instantiate a Linac

        self._sc = LinacScan(self._linac)

    def testScanParams(self):
        self._sc.add_param("param1", -0.1, 0.1, 5)
        self._sc.add_param("param2", -1., 1., 3)
        self._sc.add_param("param3", -1., sigma=0.01)
        with self.assertRaises(ValueError):
            self._sc.add_param("param3")

        lst = self._sc._generate_param_sequence(1)
        self.assertEqual(15, len(lst))
        self.assertEqual(3, len(lst[0]))

    def testJitterParams(self):
        n = 1000
        self._sc.add_param("param1", -0.1, sigma=0.01)
        self._sc.add_param("param2", -10., sigma=-0.1)
        lst = self._sc._generate_param_sequence(n)
        self.assertEqual(n, len(lst))
        self.assertEqual(2, len(lst[0]))

        lst1, lst2 = zip(*lst)
        self.assertLess(abs(0.01 - np.std(lst1)), 0.001)
        self.assertLess(abs(1 - np.std(lst2)), 0.1)
