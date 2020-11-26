import unittest
import os.path as osp

import pandas as pd
import numpy as np

from liso import (
    Phasespace,
    parse_astra_phasespace, parse_impactt_phasespace, parse_elegant_phasespace,
)
from liso.exceptions import LisoRuntimeError

SKIP_TEST = False
try:
    import sdds
except ImportError:
    SKIP_TEST = True


_ROOT_DIR = osp.dirname(osp.abspath(__file__))


class TestPhasespaceAstra(unittest.TestCase):
    def setUp(self):
        pfile = osp.join(_ROOT_DIR, "astra_output/astra.out")
        self.data = parse_astra_phasespace(pfile)

    def testInitialization(self):
        with self.assertRaises(TypeError):
            Phasespace(object(), 1.0)

        with self.assertRaises(ValueError):
            Phasespace(pd.DataFrame(columns=['x', 'y', 'px', 'py']), 1.0)

    def testConstructFromColumns(self):
        ps = Phasespace.from_columns(x=self.data['x'],
                                     px=self.data['px'],
                                     y=self.data['y'],
                                     py=self.data['py'],
                                     z=self.data['z'],
                                     pz=self.data['pz'],
                                     t=self.data['t'],
                                     charge=self.data.charge)

        for col in ['x', 'px', 'y', 'py', 'z', 'pz', 't']:
            np.testing.assert_array_equal(ps[col], self.data[col])
        self.assertEqual(self.data.charge, ps.charge)

    def testAccessItem(self):
        self.assertTupleEqual(('x', 'px', 'y', 'py', 'z', 'pz', 't'),
                              self.data.columns)

        for item in ['x', 'y', 'z', 'px', 'py', 'pz', 't', 'dt',
                     'p', 'xp', 'yp', 'dz', 'delta']:
            self.assertEqual(500, len(self.data[item]))

        with self.assertRaises(KeyError):
            self.data['aa']

    def testSlice(self):
        sliced = self.data.slice(stop=200)
        self.assertTrue(sliced['x'].equals(self.data['x'][:200]))
        self.assertEqual(500, len(self.data))

        sliced = self.data.slice(1, 100, 2)
        self.assertTrue(sliced['x'].equals(self.data['x'][1:100:2]))

        sliced = self.data.slice(stop=200, inplace=True)
        self.assertIs(sliced, self.data)
        self.assertEqual(200, len(self.data))

    def testAnalysis(self):
        with self.assertRaises(LisoRuntimeError):
            self.data.analyze(min_particles=int(2e5))

        # with self.assertRaisesRegex(RuntimeError, "slice"):
        #     analyze_phasespace(self.astra_data[:100], self.astra_charge, min_particles=20)

        params = self.data.analyze()

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

    def testCutHalo(self):
        self.data.cut_halo(0.1)
        params = self.data.analyze()
        self.assertEqual(params.n, 450)

    def testCutTail(self):
        self.data.cut_tail(0.1)
        params = self.data.analyze()
        self.assertEqual(params.n, 450)

    def testRotateBeam(self):
        self.data.rotate(0.1)
        params = self.data.analyze()
        self.assertEqual(params.n, 500)


class TestPhasespaceImpactt(unittest.TestCase):
    def setUp(self):
        pfile = osp.join(_ROOT_DIR, "impactt_output/impactt.out")
        self.data = parse_impactt_phasespace(pfile)
        self.data.charge = 1e-11

    def testZeroCharge(self):
        # to test not raise when charge = 0.0
        charge = self.data.charge
        self.data.charge = 0
        params = self.data.analyze()
        self.assertEqual(params.charge, 0.0)
        self.data.charge = charge

    def testAnalysis(self):
        params = self.data.analyze()

        self.assertAlmostEqual(params.charge, self.data.charge, places=4)
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


@unittest.skipIf(SKIP_TEST is True, "Failed to import library")
class TestPhasespaceElegant(unittest.TestCase):
    def setUp(self):
        pfile = osp.join(_ROOT_DIR, "elegant_output/elegant.out")
        self.data = parse_elegant_phasespace(pfile)

    def testAnalysis(self):
        params = self.data.analyze()

        self.assertAlmostEqual(params.charge, 1.0e-12, places=4)
        self.assertEqual(params.n, 500)
        self.assertAlmostEqual(params.p, 100.2111, places=4)
        self.assertAlmostEqual(params.Sdelta*1e2, 0.1359, places=4)
        self.assertAlmostEqual(params.St*1e12, 2.3475, places=4)

        self.assertAlmostEqual(params.emitx*1e6, 0.1215, places=2)
        self.assertAlmostEqual(params.emity*1e6, 0.1235, places=2)
        self.assertAlmostEqual(params.Sx*1e6, 149.1825, places=4)
        self.assertAlmostEqual(params.Sy*1e6, 149.4347, places=4)

        self.assertAlmostEqual(params.betax, 18.0618, places=4)
        self.assertAlmostEqual(params.betay, 17.8270, places=4)
        self.assertAlmostEqual(params.alphax, 21.0327, places=4)
        self.assertAlmostEqual(params.alphay, 20.7565, places=4)

        self.assertAlmostEqual(params.I_peak, 3.3560, places=4)

        self.assertAlmostEqual(params.Cx*1e6, -0.0846, places=1)
        self.assertAlmostEqual(params.Cy*1e6, -0.0164, places=2)
        self.assertAlmostEqual(params.Cxp*1e6, 0.1290, places=4)
        self.assertAlmostEqual(params.Cyp*1e6, 0.0195, places=4)
        self.assertAlmostEqual(params.Ct*1e9, 13.3910, places=4)

        self.assertAlmostEqual(params.emitx_slice*1e6, 0.1009, places=2)
        self.assertAlmostEqual(params.emity_slice*1e6, 0.1061, places=2)
        self.assertAlmostEqual(params.Sdelta_slice*1e4, 0.2098, places=4)
        self.assertAlmostEqual(params.Sdelta_un*1e4, 0.1199, places=4)


if __name__ == "__main__":
    unittest.main()
