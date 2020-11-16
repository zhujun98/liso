import unittest

import numpy as np

from liso.scan.scan_param import JitterParam, SampleParam, ScanParam


class TestScanParam(unittest.TestCase):
    def testScanParam(self):
        with self.assertRaises(ValueError):
            # num is 0
            ScanParam('param', start=-1., stop=1., num=0)

        param = ScanParam('param', start=-0.1, stop=0.1, num=5)
        np.testing.assert_array_almost_equal(
            [-0.1, -0.05, 0., 0.05, 0.1], param.generate())

        param = ScanParam('param', start=-0.1, stop=-0.1, num=5)
        np.testing.assert_array_almost_equal([-0.1] * 5, param.generate())

        # test positive sigma
        param = ScanParam('param', start=-10., stop=-10., num=1000, sigma=0.1)
        param_lst = param.generate()
        self.assertLess(abs(-10. - np.mean(param_lst)), 0.1)
        self.assertLess(abs(0.1 - np.std(param_lst)), 0.01)

        # test negative sigma
        param = ScanParam('param', start=-10., stop=-10., num=1000, sigma=-0.1)
        param_lst = param.generate()
        self.assertLess(abs(-10. - np.mean(param_lst)), 0.1)
        self.assertLess(abs(1. - np.std(param_lst)),  0.1)

        # test repeats and cycles
        param = ScanParam('param', start=-1., stop=1., num=3)
        np.testing.assert_array_equal([-1.0, -1.0, 0.0, 0.0, 1.0, 1.0],
                                      param.generate(repeats=2, cycles=1))
        np.testing.assert_array_equal([-1.0, -1.0, 0.0, 0.0, 1.0, 1.0] * 3,
                                      param.generate(repeats=2, cycles=3))

        # test repeats and cycles with jitter
        param = ScanParam('param', start=-1., stop=1., num=2, sigma=0.1)
        lst = param.generate(cycles=200, repeats=2)
        self.assertNotEqual(lst[0], lst[1])
        self.assertNotEqual(lst[3], lst[4])
        self.assertLess(abs(0.1 - np.std([lst[::4], lst[1::4]])), 0.01)
        self.assertLess(abs(0.1 - np.std([lst[2::4], lst[3::4]])), 0.01)

    def testJitterParam(self):
        param = JitterParam('param', value=-0.1)
        np.testing.assert_array_almost_equal([-0.1], param.generate())

        # test positive sigma
        param = JitterParam('param', value=1., sigma=0.1)
        param_lst = param.generate(repeats=3, cycles=200)
        self.assertEqual(600, len(param_lst))
        self.assertLess(abs(0.1 - np.std(param_lst)), 0.01)

        # test negative sigma
        param = JitterParam('param', value=10., sigma=-0.1)
        param_lst = param.generate(repeats=3, cycles=200)
        self.assertLess(abs(1 - np.std(param_lst)), 0.1)

    def testSampleParam(self):
        param = SampleParam('param', lb=-4, ub=6)
        param_lst = param.generate(repeats=3, cycles=200)
        self.assertEqual(600, len(param_lst))
        self.assertTrue(np.all(param_lst >= -4))
        self.assertTrue(np.all(param_lst < 6))
