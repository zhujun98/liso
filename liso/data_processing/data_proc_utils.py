"""
Author: Jun Zhu

"""
import numpy as np


def compute_canonical_emit(x, px):
    """ Calculate the canonical emittance.

    :param x: pandas.Series object
        Position coordinates
    :param px: pandas.Series object
        Momentum coordinates

    :return Normalized canonical emittance.
    """
    x_ave = x.mean()
    px_ave = px.mean()
    x2 = x.var(ddof=0)
    px2 = px.var(ddof=0)
    xpx = ((x - x_ave) * (px - px_ave)).mean()

    return np.sqrt(x2 * px2 - xpx ** 2)


def compute_twiss(x, dz, px, pz, gamma, backtracking=True):
    """ Calculate the Twiss parameters

    :param x: pandas.Series object
        Position coordinates.
    :param dz: pandas.Series object
        Longitudinal distance to the bunch centre.
    :param px: pandas.Series object
        Momentum coordinates
    :param pz: pandas.Series object
        Longitudinal momentum.
    :param gamma: float
        Average Lorentz factor of the bunch.
    :param backtracking: bool
        True for drifting the particles back to the longitudinal
        centroid of the bunch.

    :return sigma_x: float
        RMS transverse beam size.
    :return betax: float
        Beta function.
    :return alphax: float
        Alpha function.
    :return emitnx: float
        Normalized trace-space emittance
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


def gaussian_filter1d(x, sigma):
    """One-dimensional Gaussian filter.

    :param x: array_like
        Input array for filter
    :param sigma: int/float
        Standard deviation for Gaussian kernel.

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
