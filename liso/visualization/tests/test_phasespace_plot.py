import unittest

import pandas as pd

from liso.visualization.phasespace_plot import PhasespacePlot


class TestUtilities(unittest.TestCase):

    def testInstantiate(self):
        with self.assertRaises(TypeError):
            PhasespacePlot(pd.DataFrame())
