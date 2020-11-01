import pytest

import numpy as np
import pandas as pd

from liso import Normalizer


class TestNormalizer:
    @pytest.mark.parametrize(
        'method', ['log10', 'log', 'sqrt', 'cbrt'])
    def testNormalize(self, method):

        a_gt = 0.01 * np.arange(101) + 1.
        b_gt = 0.01 * (np.arange(101) - 80)
        c_gt = 2 * np.arange(101)
        data = pd.DataFrame({
            'A': a_gt.copy(),
            'B1': b_gt.copy(),
            'B2': b_gt.copy(),
            'C': c_gt.copy(),
        })

        minmax = {
            'A': [a_gt.min(), a_gt.max()],
        }

        b_shift = 10
        transform = {
            'A': method,
            'B1': (method, b_shift),
            'B2': (method, b_shift, True),
        }
        transformer = Normalizer._transformers[method]

        # test construction

        minmax['AA'] = [1, 2, 3]
        with pytest.raises(ValueError):
            Normalizer(data, minmax=minmax)
        del minmax['AA']

        minmax['BB'] = [2, 2]
        with pytest.raises(ValueError):
            Normalizer(data, minmax=minmax)
        del minmax['BB']

        norm = Normalizer(data, minmax=minmax, transform=transform)

        lb, ub = transformer(minmax['A'][0]), transformer(minmax['A'][1])
        np.testing.assert_array_almost_equal(
            2 * (transformer(a_gt, 0) - lb) / (ub - lb) - 1, data['A'])

        lb, ub = transformer(b_gt.min(), b_shift), transformer(b_gt.max(), b_shift)
        np.testing.assert_array_almost_equal(
            2 * (transformer(b_gt, b_shift) - lb) / (ub - lb) - 1, data['B1'])

        lb, ub = transformer(-b_gt.max(), b_shift), transformer(-b_gt.min(), b_shift)
        np.testing.assert_array_almost_equal(
            2 * (transformer(-b_gt, b_shift) - lb) / (ub - lb) - 1, data['B2'])

        np.testing.assert_array_almost_equal(0.01 * (c_gt - 100), data['C'])

        # test unnormalize

        norm.unnormalize(data)
        np.testing.assert_array_almost_equal(a_gt, data['A'])
        np.testing.assert_array_almost_equal(b_gt, data['B1'])
        np.testing.assert_array_almost_equal(b_gt, data['B2'])
        np.testing.assert_array_almost_equal(c_gt, data['C'])

        new_data = {
            'D': np.array([0.2, -0.2]),
        }
        with pytest.raises(KeyError, match="D is not found"):
            norm.unnormalize(new_data)
