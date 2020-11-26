"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import numpy as np
from scipy.ndimage.filters import gaussian_filter


def compute_canonical_emit(x, px):
    """ Calculate the canonical emittance.

    :param pandas.Series x: position coordinates
    :param pandas.Series px: momentum coordinates

    :return Normalized canonical emittance.
    """
    x_ave = x.mean()
    px_ave = px.mean()
    x2 = x.var(ddof=0)
    px2 = px.var(ddof=0)
    xpx = ((x - x_ave) * (px - px_ave)).mean()

    return np.sqrt(x2 * px2 - xpx ** 2)


def compute_twiss(x, dz, px, pz, gamma, backtracking=True):
    """Calculate the Twiss parameters

    Note: In the calculation (except the canonical emittance), the
    particles are drifted back to the center of the bunch without
    considering the collective effects!!!

    :param pandas.Series x: position coordinates.
    :param pandas.Series dz: longitudinal distance to the bunch centre.
    :param pandas.Series px: momentum coordinates
    :param pandas.Series pz: longitudinal momentum.
    :param float gamma: average Lorentz factor of the bunch.
    :param bool backtracking: True for drifting the particles back to
        the longitudinal centroid of the bunch.

    :return float sigma_x: RMS transverse beam size.
    :return float betax: beta function.
    :return float alphax: alpha function.
    :return float emitnx: normalized trace-space emittance
    """
    beta = np.sqrt(1 - 1 / gamma ** 2)

    xp = px / pz
    x_new = x - dz * xp

    x_ave = x_new.mean()
    xp_ave = xp.mean()
    x2 = x_new.var(ddof=0)
    xp2 = xp.var(ddof=0)
    xxp = ((x_new - x_ave) * (xp - xp_ave)).mean()

    emitx = np.sqrt(x2 * xp2 - xxp ** 2)
    emitnx = emitx * beta * gamma
    sigma_x = np.sqrt(x2)
    betax = x2 / emitx
    alphax = -1 * xxp / emitx

    return sigma_x, betax, alphax, emitnx


def compute_current_profile(t, n_bins, charge):
    """Calculate the current profile.

    :param array-like t: an array of t for each particle.
    :param int n_bins: number of current bins.
    :param float charge: total bunch charge (in C).
    """
    counts, edges = np.histogram(t, bins=n_bins)
    step_size = edges[1] - edges[0]
    centers = edges[:-1] + step_size / 2

    currents = counts * charge / (len(t) * step_size)
    return currents, centers


def gaussian_filter1d(x, sigma):
    """One-dimensional Gaussian filter.

    :param array-like x: input array for filter
    :param int/float sigma: standard deviation for Gaussian kernel.

    :return: Filtered x.
    """
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(4.0*sd + 0.5)
    weights = [0.0]*(2*lw + 1)
    weights[lw] = 1.0
    sum_ = 1.0
    sd *= sd
    # calculate the kernel:
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * float(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum_ += 2.0 * tmp

    weights /= sum_

    return np.convolve(x, weights, 'same')


def mesh_phasespace(x, y, *,
                    x_bins=30,
                    y_bins=30,
                    x_range=None,
                    y_range=None):
    """Return the meshed phasespace for displaying as an image.

    :param array-like x: 1D x data.
    :param array-like y: 1D y data.
    :param int x_bins: number of bins in x.
    :param int y_bins: number of bins in y.
    :param tuple x_range: binning range in x.
    :param tuple y_range: binning range in y.

    :returns numpy.ndarray counts: number of particles in the 2D meshed grid.
        Shape = (y_bins, x_bins)
    :returns numpy.ndarray x_centers: bin centers of x data. Shape = (x_bins,)
    :returns numpy.ndarray y_centers: bin centers of y data. Shape = (y_bins,)
    """
    if x_range is None:
        x_range = (x.min(), x.max())
    if y_range is None:
        y_range = (y.min(), y.max())

    counts, y_edges, x_edges = np.histogram2d(
        y, x, bins=(y_bins, x_bins), range=(y_range, x_range))
    x_centers = (x_edges[1:] + x_edges[0:-1]) / 2
    y_centers = (y_edges[1:] + y_edges[0:-1]) / 2

    return counts, x_centers, y_centers


def phasespace_density(x, y, *,
                       samples=20000, x_bins=30, y_bins=30, sigma=None):
    """Return the sampled phasespace with probability densities.

    :param array-like x: 1D x data.
    :param array-like y: 1D y data.
    :param int samples: number of data points to be sampled.
    :param int x_bins: number of bins in x.
    :param int y_bins: number of bins in y.
    :param numeric sigma: standard deviation of Gaussian kernel of the
        Gaussian filter.

    :returns numpy.ndarray z: Normalized density at each sample point.
        Shape = (samples,)
    :returns numpy.ndarray x_sample: sampled x data. Shape = (samples,)
    :returns numpy.ndarray y_sample: sampled y data. Shape = (samples,)
    """
    counts, x_edges, y_edges = np.histogram2d(x, y, bins=(x_bins, y_bins))

    if sigma is not None:
        counts = gaussian_filter(counts, sigma=sigma)

    # sample phasespace
    if samples >= len(x):
        x_sample, y_sample = x, y
    else:
        x_sample = np.random.choice(x, size=samples)
        y_sample = np.random.choice(y, size=samples)

    px = np.digitize(x_sample, x_edges, right=True)
    py = np.digitize(y_sample, y_edges, right=True)

    px[px == x_bins] = x_bins - 1
    py[py == y_bins] = y_bins - 1
    z = counts[px, py]
    z /= z.sum()

    return z, x_sample, y_sample
