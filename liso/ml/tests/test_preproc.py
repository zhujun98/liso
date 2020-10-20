import unittest

import numpy as np
import pandas as pd

from liso.ml import Normalizer


class TestNormalizer(unittest.TestCase):
    def testNormalize(self):
        minmax = {
            'A': [0, 100],
        }
        a_gt = np.arange(101)
        c_gt = 2 * np.arange(101)
        data = pd.DataFrame({
            'A': a_gt.copy(),
            'C': c_gt.copy(),
        })

        # test construction

        minmax['AA'] = [1, 2, 3]
        with self.assertRaises(ValueError):
            Normalizer(data, minmax=minmax)
        del minmax['AA']

        norm = Normalizer(data, minmax=minmax)

        np.testing.assert_array_almost_equal(0.02 * (a_gt - 50), data['A'])
        np.testing.assert_array_almost_equal(0.01 * (c_gt - 100), data['C'])

        # test normalize

        new_data = {
            'A': np.array([5, 10]),
        }
        norm.normalize(new_data)
        np.testing.assert_array_almost_equal([-0.9, -0.8], new_data['A'])

        new_data = {
            'D': np.array([2.2, 1.8]),
        }
        with self.assertRaisesRegex(KeyError, "D is not found"):
            norm.normalize(new_data)

        # test unnormalize

        new_data = {
            'A': np.array([-0.9, -0.8]),
        }
        norm.unnormalize(new_data)
        np.testing.assert_array_almost_equal([5, 10], new_data['A'])

        new_data = {
            'D': np.array([0.2, -0.2]),
        }
        with self.assertRaisesRegex(KeyError, "D is not found"):
            norm.normalize(new_data)
