import pytest

import pandas as pd
import numpy as np

from liso import (
    phasespace_density, mesh_phasespace
)


@pytest.mark.parametrize(
    "x, y",
    [(np.random.randn(1000), np.random.randn(1000)),
     (pd.Series(np.random.randn(1000)), pd.Series(np.random.randn(1000)))])
def test_mesh_phasespace(x, y):
    intensity, xc, yc = mesh_phasespace(x, y)
    assert intensity.shape == (30, 30)

    xe_gt = np.linspace(x.min(), x.max(), 31)
    xc_gt = (xe_gt[1:] + xe_gt[:-1]) / 2.
    np.testing.assert_array_almost_equal(xc_gt, xc)
    ye_gt = np.linspace(y.min(), y.max(), 31)
    yc_gt = (ye_gt[1:] + ye_gt[:-1]) / 2.
    np.testing.assert_array_almost_equal(yc_gt, yc)

    intensity, xc, yc = mesh_phasespace(
        x, y, x_bins=20, y_bins=30, x_range=(-2, 2), y_range=(0, 2))
    assert intensity.shape == (30, 20)

    xe_gt = np.linspace(-2, 2, 21)
    xc_gt = (xe_gt[1:] + xe_gt[:-1]) / 2.
    np.testing.assert_array_almost_equal(xc_gt, xc)
    ye_gt = np.linspace(0, 2, 31)
    yc_gt = (ye_gt[1:] + ye_gt[:-1]) / 2.
    np.testing.assert_array_almost_equal(yc_gt, yc)


@pytest.mark.parametrize(
    "x, y",
    [(np.random.randn(30000), np.random.randn(30000)),
     (pd.Series(np.random.randn(30000)), pd.Series(np.random.randn(30000)))])
def test_phasespace_density1(x, y):
    z, x_sample, y_sample = phasespace_density(
        x, y, samples=25000, x_bins=10, y_bins=20, sigma=0.1)

    assert 1.0 == pytest.approx(z.sum())
    assert 25000 == len(x_sample)
    assert 25000 == len(y_sample)


def test_phasespace_density2():
    x = [1, 2, 3]
    y = [4, 5, 6]
    z, x_sample, y_sample = phasespace_density(x, y, x_bins=10, y_bins=10)

    np.testing.assert_array_almost_equal([1/3] * 3, z)
    np.testing.assert_array_equal(x, x_sample)
    np.testing.assert_array_equal(y, y_sample)
