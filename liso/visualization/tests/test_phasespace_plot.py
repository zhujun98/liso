import unittest
import os.path as osp

from liso import parse_astra_phasespace, PhasespacePlot


_ROOT_DIR = osp.dirname(osp.abspath(__file__))


class TestUtilities(unittest.TestCase):

    def setUp(self):
        pfile = osp.join(_ROOT_DIR, "../../proc/tests/astra_output/astra.out")
        self._data = parse_astra_phasespace(pfile)

        self.psp = PhasespacePlot(self._data)

    def testPlot(self):
        self.psp.plot('x', 'y')
        self.psp.plot('t', 'p')
        self.psp.plot('t', 'p', show_current=True, show_parameters=False)

    def testImshow(self):
        self.psp.imshow('x', 'y')
        self.psp.imshow('x', 'y', flip_origin=False)
