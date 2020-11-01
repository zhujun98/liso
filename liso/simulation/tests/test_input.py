import unittest
import os.path as osp
import tempfile

import numpy as np

from liso.proc import (
    parse_astra_phasespace, parse_impactt_phasespace, parse_elegant_phasespace,
)
from liso.simulation import ParticleFileGenerator
from liso.simulation.input import (
    AstraInputGenerator, ImpacttInputGenerator, ElegantInputGenerator
)

SKIP_TEST = False
try:
    import sdds
except ImportError:
    SKIP_TEST = True

_ROOT_DIR = osp.dirname(osp.abspath(__file__))


class TestParticleFileGenerator(unittest.TestCase):
    def testAstraCathode(self):
        n = 2000
        charge = 1e-9
        gen = ParticleFileGenerator(n, charge, cathode=True, seed=42,
                                    dist_x='uniform', sig_x=1e-3,
                                    dist_z='gaussian', sig_z=1e-12,
                                    ek=0.55)

        with tempfile.NamedTemporaryFile('w') as file:
            gen.to_astra(file.name)

            data = np.loadtxt(file.name)

            self.assertEqual(0, data[0, 0])  # x ref
            self.assertEqual(0, data[0, 1])  # y ref
            self.assertTrue(np.all(data[:, 2] == 0.))  # index
            self.assertEqual(0, data[0, 3])  # px ref
            self.assertEqual(0, data[0, 4])  # py ref

            p_ref = data[0, 5]
            for i in [1, 101, 1001, -1]:
                self.assertAlmostEqual(
                    p_ref,
                    np.sqrt(data[i, 3] ** 2 + data[i, 4] ** 2 + (p_ref + data[i, 5]) ** 2),
                    places=4
                )

            self.assertTrue(np.all(data[:, 7] == -1e9 * charge / n))  # index
            self.assertTrue(np.all(data[:, 8] == 1))  # index
            self.assertTrue(np.all(data[:, 9] == -1))  # flag

    def testAstraFromPhasespace(self):
        pfile = osp.join(
            _ROOT_DIR, "../../proc/tests/astra_output/astra.out")
        ps = parse_astra_phasespace(pfile)
        param_gt = ps.analyze()

        gen = ParticleFileGenerator.from_phasespace(ps)
        with tempfile.NamedTemporaryFile('w') as file:
            gen.to_astra(file.name)

            param = parse_astra_phasespace(file.name).analyze()

            for attr in ['n', 'q', 'Sz', 'betay', 'emity']:
                self.assertAlmostEqual(getattr(param_gt, attr), getattr(param, attr), places=4)
            self.assertAlmostEqual(param_gt.p, param.p, places=3)

    def testImpacttFromPhasespace(self):
        pfile = osp.join(
            _ROOT_DIR, "../../proc/tests/impactt_output/impactt.out")
        ps = parse_impactt_phasespace(pfile)
        param_gt = ps.analyze()

        gen = ParticleFileGenerator.from_phasespace(ps)
        with tempfile.NamedTemporaryFile('w') as file:
            gen.to_impactt(file.name)

            param = parse_impactt_phasespace(file.name).analyze()

            for attr in ['n', 'q', 'gamma', 'St', 'betax', 'emitx']:
                self.assertAlmostEqual(getattr(param_gt, attr), getattr(param, attr))

    @unittest.skipIf(SKIP_TEST is True, "Failed to import library")
    def testElegantFromPhasespace(self):
        pfile = osp.join(
            _ROOT_DIR, "../../proc/tests/elegant_output/elegant.out")
        ps = parse_elegant_phasespace(pfile)
        param_gt = ps.analyze()

        gen = ParticleFileGenerator.from_phasespace(ps)
        with tempfile.NamedTemporaryFile('w') as file:
            gen.to_elegant(file.name)

            param = parse_elegant_phasespace(file.name).analyze()

            for attr in ['n', 'q', 'gamma', 'St', 'betax', 'emitx']:
                self.assertAlmostEqual(getattr(param_gt, attr), getattr(param, attr))


class TestAstraInputGenerator(unittest.TestCase):
    def setUp(self):
        self._gen = AstraInputGenerator(osp.join(_ROOT_DIR, "./injector.in.000"))

    def test_raises(self):
        mapping = {'gun_gradient': 10, 'gun_phase0': 20}
        # 'gun_phase' not in the mapping
        with self.assertRaisesRegex(KeyError, "No mapping"):
            self._gen.update(mapping)

        # 'tws_gradient' is redundant
        mapping = {'gun_gradient': 10, 'gun_phase': 20, 'tws_gradient': 30}
        with self.assertRaisesRegex(KeyError, "not found"):
            self._gen.update(mapping)

    def test_not_raise(self):
        mapping = {'gun_gradient': 10, 'gun_phase': 20}
        self._gen.update(mapping)
        with tempfile.NamedTemporaryFile('w') as file:
            self._gen.write(file.name)


class TestImpacttInputGenerator(unittest.TestCase):
    def setUp(self):
        self._gen = ImpacttInputGenerator(osp.join(_ROOT_DIR, "./ImpactT.in.000"))

    def test_raises(self):
        mapping = {'MQZM1_G': 10, 'MQZM3_G': 20}
        with self.assertRaisesRegex(KeyError, "No mapping"):
            self._gen.update(mapping)

        mapping = {'MQZM1_G': 10, 'MQZM2_G': 20, 'MQZM3_G': 30}
        with self.assertRaisesRegex(KeyError, "not found"):
            self._gen.update(mapping)

    def test_not_raise(self):
        mapping = {'MQZM1_G': 10, 'MQZM2_G': 20}
        self._gen.update(mapping)
        with tempfile.NamedTemporaryFile('w') as file:
            self._gen.write(file.name)


class TestElegantInputGenerator(unittest.TestCase):
    def setUp(self):
        self._gen = ElegantInputGenerator(osp.join(_ROOT_DIR, "./elegant.ele.000"))

    def test_not_raise(self):
        mapping = {}
        self._gen.update(mapping)
        with tempfile.NamedTemporaryFile('w') as file:
            self._gen.write(file.name)
