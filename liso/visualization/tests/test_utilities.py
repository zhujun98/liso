import unittest

import numpy as np
import pandas as pd

from liso.visualization.vis_utils import (
    fast_sample_data, get_default_unit, get_label, get_html_label, get_line_column_by_name,
    get_phasespace_column_by_name, get_unit_label_and_scale, sample_data
)


class TestUtilities(unittest.TestCase):
    def testFastSampleData(self):
        x = pd.Series(np.arange(1000))
        y = pd.Series(np.arange(1000) + 100)

        xs, ys = fast_sample_data(x, y, n=10)
        self.assertEqual(10, len(xs))
        self.assertEqual(10, len(ys))

        xs, ys = fast_sample_data(x, y, n=2000)
        self.assertEqual(1000, len(xs))
        self.assertEqual(1000, len(ys))

    def testSampleData(self):
        pass

    def testGetlabel(self):
        self.assertEqual(get_label('gamma'), get_label('Gamma'))

    def testGetHtmlLabel(self):
        self.assertEqual(get_html_label('sx'), get_html_label('sX'))

    def testGetDefaultUnit(self):
        self.assertEqual('mm', get_default_unit('x'))
        self.assertEqual('m', get_default_unit('z'))
        self.assertEqual('um', get_default_unit('emitx'))

    def testGetUnitLabelAndScale(self):
        self.assertTupleEqual(('(kA)', 1.e-3), get_unit_label_and_scale('kA'))
        self.assertTupleEqual(('(MeV)', 1.e-6), get_unit_label_and_scale('mev'))

    def testGetPhasespaceColumnByName(self):
        pass

    def testGetLineColumnByName(self):
        pass
