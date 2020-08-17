import unittest
import tempfile

import numpy as np

from liso.simulation import ParticleFileGenerator


class TestParticleFileGenerator(unittest.TestCase):
    def testAstraCachode(self):
        n = 2000
        charge = 1e-9
        gen = ParticleFileGenerator(n, charge, cathode=True, seed=42,
                                    dist_x='uniform', sig_x=1e-3,
                                    dist_z='gaussian', sig_z=1e-12,
                                    ek=0.55)

        with tempfile.NamedTemporaryFile('w') as file:
            gen.toAstra(file.name)

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
