"""
Unittest for Data Analysis

Author: Jun Zhu, zhujun981661@gmail.com
"""
import unittest

import numpy as np

from liso.data_processing import (
    analyze_line, parse_astra_line, parse_impactt_line,
)


class TestAnalyzeLine(unittest.TestCase):

    def testAstra(self):
        astra_data = parse_astra_line("astra_output/injector")

        with self.assertRaises(RuntimeError):
            analyze_line([1, 2], max)

        params = analyze_line(astra_data, np.max)
        params = analyze_line(astra_data, np.std)

    def testImpact(self):
        impactt_data = parse_impactt_line("impactt_output/fort")

        with self.assertRaises(RuntimeError):
            analyze_line([1, 2], max)

        params = analyze_line(impactt_data, np.min)
        params = analyze_line(impactt_data, np.var)


if __name__ == "__main__":
    unittest.main()
