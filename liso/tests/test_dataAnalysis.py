"""
Unittest for Data Analysis

Author: Jun Zhu, zhujun981661@gmail.com
"""
import unittest
import os

import numpy as np

from liso.data_processing import (
    analyze_beam, analyze_line, parse_astra_line, parse_impactt_line,
    parse_astra_phasespace, parse_impactt_phasespace, tailor_beam
)

test_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'files4test'
))


class TestAnalyzeBeam(unittest.TestCase):
    def setUp(self):
        pfile = os.path.join(test_path, "impactt_output/impactt.out")
        self.impactt_data, _ = parse_impactt_phasespace(pfile)
        self.impactt_charge = 1e-11

        pfile = os.path.join(test_path, "astra_output/astra.out")
        self.astra_data, self.astra_charge = parse_astra_phasespace(pfile)

    def testAstra(self):
        with self.assertRaises(RuntimeError):
            analyze_beam(self.astra_data[:19], self.astra_charge, min_particles=20)

        # with self.assertRaisesRegex(RuntimeError, "slice"):
        #     analyze_beam(self.astra_data[:100], self.astra_charge, min_particles=20)

        params = analyze_beam(self.astra_data, self.astra_charge)

        self.assertAlmostEqual(params.charge, 1.0e-12, places=4)
        self.assertEqual(params.n, 500)
        self.assertAlmostEqual(params.p, 100.2111, places=4)
        self.assertAlmostEqual(params.Sdelta*1e2, 0.1359, places=4)
        self.assertAlmostEqual(params.St*1e12, 2.3475, places=4)

        self.assertAlmostEqual(params.emitx*1e6, 0.1215, places=4)
        self.assertAlmostEqual(params.emity*1e6, 0.1235, places=4)
        self.assertAlmostEqual(params.Sx*1e6, 149.1825, places=4)
        self.assertAlmostEqual(params.Sy*1e6, 149.4347, places=4)

        self.assertAlmostEqual(params.betax, 18.0618, places=4)
        self.assertAlmostEqual(params.betay, 17.8270, places=4)
        self.assertAlmostEqual(params.alphax, 21.0327, places=4)
        self.assertAlmostEqual(params.alphay, 20.7565, places=4)

        self.assertAlmostEqual(params.I_peak, 3.3560, places=4)

        self.assertAlmostEqual(params.Cx*1e6, -0.0846, places=4)
        self.assertAlmostEqual(params.Cy*1e6, -0.0164, places=4)
        self.assertAlmostEqual(params.Cxp*1e6, 0.1290, places=4)
        self.assertAlmostEqual(params.Cyp*1e6, 0.0195, places=4)
        self.assertAlmostEqual(params.Ct*1e9, 13.3910, places=4)

        self.assertAlmostEqual(params.emitx_slice*1e6, 0.1009, places=4)
        self.assertAlmostEqual(params.emity_slice*1e6, 0.1061, places=4)
        self.assertAlmostEqual(params.Sdelta_slice*1e4, 0.2098, places=4)
        self.assertAlmostEqual(params.Sdelta_un*1e4, 0.1199, places=4)

    def testImpactt(self):
        params = analyze_beam(self.impactt_data, self.impactt_charge)

        self.assertAlmostEqual(params.charge, self.impactt_charge, places=4)
        self.assertEqual(params.n, 500)
        self.assertAlmostEqual(params.p, 100.2111, places=4)
        self.assertAlmostEqual(params.Sdelta*1e2, 0.1359, places=4)
        self.assertAlmostEqual(params.St*1e12, 2.3476, places=4)

        self.assertAlmostEqual(params.emitx*1e6, 0.1214, places=4)
        self.assertAlmostEqual(params.emity*1e6, 0.1227, places=4)
        self.assertAlmostEqual(params.Sx*1e6, 4.9346, places=4)
        self.assertAlmostEqual(params.Sy*1e6, 116.0314, places=4)

        self.assertAlmostEqual(params.betax, 0.0197, places=4)
        self.assertAlmostEqual(params.betay, 10.7403, places=4)
        self.assertAlmostEqual(params.alphax, -0.0662, places=4)
        self.assertAlmostEqual(params.alphay, -31.1515, places=4)

        self.assertAlmostEqual(params.I_peak, 1.7029, places=4)

        self.assertAlmostEqual(params.Cx*1e6, 0.0225, places=4)
        self.assertAlmostEqual(params.Cy*1e6, 0.0037, places=4)
        self.assertAlmostEqual(params.Cxp*1e6, 0.1213, places=4)
        self.assertAlmostEqual(params.Cyp*1e6, 0.0110, places=4)
        self.assertAlmostEqual(params.Ct*1e9, 0.0000, places=4)

        self.assertAlmostEqual(params.emitx_slice*1e6, 0.1009, places=4)
        self.assertAlmostEqual(params.emity_slice*1e6, 0.1061, places=4)
        self.assertAlmostEqual(params.Sdelta_slice*1e4, 0.2108, places=4)
        self.assertAlmostEqual(params.Sdelta_un*1e4, 0.1207, places=4)

    def testZeroCharge(self):
        # to test not raise when charge = 0.0
        params = analyze_beam(self.impactt_data, 0.0)
        self.assertEqual(params.charge, 0.0)

    def testCutHalo(self):
        params = analyze_beam(tailor_beam(self.astra_data, halo=0.1),
                              self.astra_charge)
        self.assertEqual(params.n, 450)

    def testCutTail(self):
        params = analyze_beam(tailor_beam(self.astra_data, tail=0.1),
                              self.astra_charge)
        self.assertEqual(params.n, 450)

    def testRotateBeam(self):
        params = analyze_beam(tailor_beam(self.astra_data, rotation=0.1),
                              self.astra_charge)
        self.assertEqual(params.n, 500)

    def testTailorBeam(self):
        params = analyze_beam(tailor_beam(self.astra_data, halo=0.1, tail=0.2,
                                          rotation=0.1),
                              self.astra_charge)
        self.assertEqual(params.n, 360)


class TestAnalyzeLine(unittest.TestCase):

    def testAstra(self):
        astra_data = parse_astra_line(
            os.path.join(test_path, "astra_output/injector"))

        with self.assertRaises(RuntimeError):
            analyze_line([1, 2], max)

        params = analyze_line(astra_data, np.max)
        params = analyze_line(astra_data, np.std)

    def testImpact(self):
        impactt_data = parse_impactt_line(
            os.path.join(test_path, "impactt_output/fort"))

        with self.assertRaises(RuntimeError):
            analyze_line([1, 2], max)

        params = analyze_line(impactt_data, np.min)
        params = analyze_line(impactt_data, np.var)


if __name__ == "__main__":
    unittest.main()
