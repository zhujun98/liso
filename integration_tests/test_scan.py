"""
Unittest of parameter_scan study with ASTRA.

Author: Jun Zhu, zhujun981661@gmail.com
"""
import os
import glob
import unittest

import numpy as np

from liso import Linac, LinacScan
from liso.logging import logger
logger.setLevel('CRITICAL')

test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'scan'))


class TestLinacScan(unittest.TestCase):
    def setUp(self):
        linac = Linac()
        linac.add_beamline('astra',
                           name='gun',
                           fin=os.path.join(test_path, 'injector.in'),
                           template=os.path.join(test_path, 'injector.in.000'),
                           pout='injector.0150.001')

        # set an parameter_scan problem
        self._sc = LinacScan(linac)

    def tearDown(self):
        for file in glob.glob(os.path.join(test_path, "injector.*.001")):
            os.remove(file)
        os.remove(os.path.join(test_path, "injector.in"))

    def testGeneral(self):
        n = 5
        self._sc.add_param('gun_gradient', np.linspace(130 - 0.06, 130 + 0.06, n))
        with self.assertRaisesRegex(ValueError, "different lengths"):
            self._sc.add_param('gun_phase', np.linspace(-0.02, 0.02, n + 1))
        self._sc.add_param('gun_phase', np.linspace(-0.02, 0.02, n))
        self._sc.scan()


if __name__ == "__main__":
    unittest.main()
