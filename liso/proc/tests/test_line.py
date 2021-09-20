"""
Unittest for Data Analysis

Author: Jun Zhu, zhujun981661@gmail.com
"""
import unittest
import os.path as osp

import numpy as np

from liso import (
    analyze_line, parse_astra_line, parse_impactt_line,
)
from liso.exceptions import LisoRuntimeError


_ROOT_DIR = osp.dirname(osp.abspath(__file__))


class TestAnalyzeLine(unittest.TestCase):

    def testAstra(self):
        astra_data = parse_astra_line(osp.join(_ROOT_DIR, "astra_output/injector"))

        with self.assertRaises(LisoRuntimeError):
            analyze_line([1, 2], max)

        analyze_line(astra_data, np.max)
        analyze_line(astra_data, np.std)

    def testImpact(self):
        impactt_data = parse_impactt_line(osp.join(_ROOT_DIR, "impactt_output/fort"))

        with self.assertRaises(LisoRuntimeError):
            analyze_line([1, 2], max)

        analyze_line(impactt_data, np.min)
        analyze_line(impactt_data, np.var)
