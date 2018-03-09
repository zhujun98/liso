"""
Author: Jun Zhu

"""
import numpy as np
from .beam_parameters import BeamParameters
from .line_parameters import LineParameters, Stats
from ..exceptions import *


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


def analyze_beam(data, charge, *,
                 cut_halo=0.0,
                 cut_tail=0.0,
                 current_bins='auto',
                 filter_size=1,
                 slice_percent=0.1,
                 slice_with_peak_current=True,
                 minimum_particles=5):
    """Calculate beam parameters.

    :param data: Pandas.DataFrame
        Particle data.
    :param cut_halo: None/float
        Percentage of particles to be removed based on their
        transverse distance to the bunch centroid. Applied
        before tail cutting.
    :param cut_tail: None/float
        Percentage of particles to be removed in the tail.
    :param current_bins: int/'auto'
        No. of bins to calculate the current profile.
    :param filter_size: int/float
        Standard deviation of the Gaussian kernel of the 1D Gaussian
        filter used for current profile calculation.
    :param slice_percent: float
        Percent of the slice bunch length to the total bunch length.
    :param slice_with_peak_current: Boolean
        True for calculating slice properties of the slice with peak
        current; False for calculating slice properties of the slice
        in the center of the bunch.
    :param minimum_particles: int
        If the number of particles is less than "minimum_particles", it
        will throw an exception.

    :return: BeamParameters object.
        Beam parameters.
    """
    params = BeamParameters()

    n0 = len(data)
    params.n = n0  # Number of particles after processing
    params.q = charge / n0  # charge per particle

    # Cut the halo of the bunch
    params.n = int(params.n * (1 - cut_halo))

    # TODO: avoid creating a column and then deleting it
    data['r'] = np.sqrt(data['x'] ** 2 + data['y'] ** 2)
    data = data.reindex(data['r'].sort_values(ascending=True).index)
    data = data[:params.n]
    del data['r']

    # Cut the tail of the bunch
    params.n = int(params.n * (1 - cut_tail))
    data['t'] -= data['t'].median()  # User median() to deal with extreme outliers
    data = data.reindex(data['t'].abs().sort_values(ascending=True).index)
    data = data[:params.n]
    data['t'] -= data['t'].mean()

    # Too few particles may cause error during the following
    # calculation, e.g. negative value in sqrt.
    if params.n < minimum_particles:
        raise TooFewOutputParticlesError("Too few particles ({}) in the data!".
                                         format(params.n))

    p = np.sqrt(data['pz'] ** 2 + data['px'] ** 2 + data['py'] ** 2)

    p_ave = p.mean()
    dp = (p - p_ave) / p_ave
    dz = data['z'] - data['z'].mean()

    params.p = p_ave
    params.gamma = np.sqrt(p_ave ** 2 + 1)
    params.chirp = -1 * dp.cov(dz) / dz.var(ddof=0)
    params.Sdelta = p.std(ddof=0) / p_ave
    params.St = data['t'].std(ddof=0)
    params.Sz = data['z'].std(ddof=0)

    # The current profile calculation is included here but not in
    # the plot function. If it is included in the plot function, the
    # printout parameters might differ from the plot, which could
    # cause confusion.
    counts, edges = np.histogram(data['t'], bins=current_bins)
    step_size = edges[1] - edges[0]
    centers = edges[:-1] + step_size / 2
    current = counts / float(len(data)) * params.charge / step_size
    current = gaussian_filter1d(current, sigma=filter_size)
    params.I_peak = current.max()
    params.current_dist = [centers, current]

    params.emitx = compute_canonical_emit(data['x'], data['px'])

    params.Sx, params.betax, params.alphax, params.emitx_tr \
        = compute_twiss(data['x'], dz, data['px'], data['pz'], params.gamma)

    params.emity = compute_canonical_emit(data['y'], data['py'])

    params.Sy, params.betay, params.alphay, params.emity_tr \
        = compute_twiss(data['y'], dz, data['py'], data['pz'], params.gamma)

    params.Cx = data['x'].mean()
    params.Cy = data['y'].mean()
    params.Cxp = (data['px'] / data['pz']).mean()
    params.Cyp = (data['py'] / data['pz']).mean()
    params.Ct = data['t'].mean()

    # Calculate the slice parameters
    sorted_data = data.reindex(data['t'].abs().sort_values(ascending=True).index)

    if slice_with_peak_current is True:
        Ct_slice = centers[np.argmax(current)]
    else:
        Ct_slice = params.Ct

    dt_slice = 4 * params.St * slice_percent  # assume 4-sigma full bunch length
    slice_data = sorted_data[(sorted_data.t > Ct_slice - dt_slice / 2) &
                             (sorted_data.t < Ct_slice + dt_slice / 2)]

    if len(slice_data) < minimum_particles:
        raise TooFewOutputParticlesError("Too few particles ({}) in the slice data!".
                                         format(len(slice_data)))

    p_slice = np.sqrt(slice_data['pz'] ** 2 + slice_data['px'] ** 2 + slice_data['py'] ** 2)

    params.emitx_slice = compute_canonical_emit(slice_data.x, slice_data.px)
    params.emity_slice = compute_canonical_emit(slice_data.y, slice_data.py)
    params.Sdelta_slice = p_slice.std(ddof=0) / p_slice.mean()
    params.dt_slice = slice_data.t.max() - slice_data.t.min()

    # The output will be different from the output by msddsplot
    # because the slightly different No of particles sliced. It
    # affects the correlation calculation since the value is
    # already very close to 1.
    params.Sdelta_un = params.Sdelta_slice * np.sqrt(1 - (slice_data['t'].corr(p_slice)) ** 2)

    return params


def analyze_line(data, zlim):
    """Calculate line parameters.

    :param zlim: tuple, (z_min, z_max)
        Range of the z coordinate.

    :return: A LineParameter object.
    """
    # Slice data in the range of self.z_lim
    z_max = min(data['z'].max(), zlim[1])
    z_min = max(data['z'].min(), zlim[0])

    i_min = 0
    i_max = len(data['z'])
    for i in range(i_max):
        if z_min <= data['z'][i]:
            i_min = i
            break

    for i in range(i_max):
        if z_max < data['z'][i]:
            i_max = i - 1
            break
    data = data.ix[i_min:i_max]

    params = LineParameters()
    for key, value in params.__dict__.items():
        if isinstance(value, Stats):
            value.update(data[key])

    return params
