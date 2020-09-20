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

    def testItertools(self):
        param = ScanParam('param', -1., 1., 3)

        self.assertListEqual([-1.0, -1.0, 0.0, 0.0, 1.0, 1.0], param.repeat(2))
        self.assertListEqual([-1.0, -1.0, 0.0, 0.0, 1.0, 1.0] * 3, param.cycle(3, 2))

        param = ScanParam('param', 1., sigma=0.1)
        self.assertLess(abs(0.1 - np.std(param.cycle(200, repeat=3))), 0.01)

        param = ScanParam('param', -1., 1., 2, sigma=0.1)
        lst = param.cycle(200, repeat=2)
        self.assertNotEqual(lst[0], lst[1])
        self.assertNotEqual(lst[3], lst[4])
        self.assertLess(abs(0.1 - np.std(lst[::4] + lst[1::4])), 0.01)
        self.assertLess(abs(0.1 - np.std(lst[2::4] + lst[3::4])), 0.01)
