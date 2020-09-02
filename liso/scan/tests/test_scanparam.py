import unittest

import numpy as np

from liso.scan.scan_param import ScanParam


class TestScanParam(unittest.TestCase):
    def testGeneral(self):
        with self.assertRaises(ValueError):
            ScanParam('param', -1., 1., 0)

        param1 = ScanParam('param1', -0.1, 0.1, 5)
        np.testing.assert_array_almost_equal(
            [-0.1, -0.05, 0., 0.05, 0.1], list(iter(param1)))

        param2 = ScanParam('param2', -0.1, None, 5)
        np.testing.assert_array_almost_equal([-0.1], list(iter(param2)))

        param3 = ScanParam('param3', -0.1, -0.1, 5)
        np.testing.assert_array_almost_equal([-0.1] * 5, list(iter(param3)))

        param4 = ScanParam('param4', -10., -10., 1000, sigma=0.1)
        param4_lst = list(iter(param4))
        self.assertTrue(abs(-10. - np.mean(param4_lst)) < 0.1)
        self.assertTrue(abs(0.1 - np.std(param4_lst)) < 0.01)

        param5 = ScanParam('param5', -10., -10., 1000, sigma=-0.1)
        param5_lst = list(iter(param5))
        self.assertTrue(abs(-10. - np.mean(param5_lst)) < 0.1)
        self.assertTrue(abs(1. - np.std(param5_lst)) < 0.1)

        with self.assertRaises(StopIteration):
            next(param5)
        param5.reset()
        self.assertEqual(1000, len(list(iter(param5))))
