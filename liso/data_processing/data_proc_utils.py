"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import os
import numpy as np
from .beam_parameters import BeamParameters
from .line_parameters import LineParameters
from ..exceptions import *


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


def cut_halo(data, value):
    """Remove halo from a phase-space distribution.

    :param Pandas.DataFrame data: particle data.
    :param float value: percentage of particles to be removed based on
        their transverse distance to the bunch centroid. Applied
        before tail cutting.

    :return Pandas.DataFrame: Truncated data.
    """
    if isinstance(value, float) and 1.0 > value > 0.0:
        n = int(len(data) * (1 - value))
        # TODO: avoid creating a column and then deleting it
        data['r'] = np.sqrt(data['x'] ** 2 + data['y'] ** 2)
        data = data.reindex(data['r'].sort_values(ascending=True).index)
        data = data[:n]
        del data['r']

    return data


def cut_tail(data, value):
    """Remove tail from a phase-space distribution.

    :param Pandas.DataFrame data: particle data.
    :param float value: percentage of particles to be removed in the tail.

    :return Pandas.DataFrame: truncated data.
    """
    if isinstance(value, float) and 1.0 > value > 0.0:
        n = int(len(data) * (1 - value))
        # TODO: avoid creating a column and then deleting it
        data['t_'] = data['t'] - data['t'].median()  # User median() to deal with extreme outliers
        data = data.reindex(data['t_'].abs().sort_values(ascending=True).index)
        data = data[:n]
        del data['t_']

    return data


def rotate(data, angle):
    """Rotate the phasespace.

    :param Pandas.DataFrame data: particle data.
    :param float angle: angle of the rotation in rad.

    :return Pandas.DataFrame: rotated data.
    """
    if angle != 0.0:
        theta = angle * np.pi / 180.0  # Convert to rad

        # x(m), px(mc), y(m), py(mc), t(s), p(mc), z(m).

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        def transformation(r, cm):
            x = r[0]
            y = r[1]
            z = r[2]
            cx = cm[0]
            cy = cm[1]
            cz = cm[2]

            x_new = cx - cx*cos_theta + x*cos_theta - cz*sin_theta + z*sin_theta
            y_new = y
            z_new = cz - cz*cos_theta + z*cos_theta + cx*sin_theta - x*sin_theta

            return [x_new, y_new, z_new]

        pos = [data['x'], data['y'], data['z']]
        cm_pos = [np.mean(data['x']), np.mean(data['y']), np.mean(data['z'])]
        mom = [data['px'], data['py'], data['pz']]
        cm_mom = [np.mean(data['px']), np.mean(data['py']),
                  np.mean(data['pz'])]

        [data['x'], data['y'], data['z']] = transformation(pos, cm_pos)
        [data['px'], data['py'], data['pz']] = transformation(mom, cm_mom)

    return data


def tailor_beam(data, *, halo=0.0, tail=0.0, rotation=0.0):
    """Tailor the beam.

    :param Pandas.DataFrame data: particle data.
    :param float halo:percentage of particles to be removed based on
        their transverse distance to the bunch centroid. Applied
        before tail cutting.
    :param float halo: percentage of particles to be removed based on
        their transverse distance to the bunch centroid. Applied
        before tail cutting.
    :param float rotation:angle of the rotation in rad.

    :return Pandas.DataFrame: tailored data.
    """
    return cut_halo(cut_tail(rotate(data, rotation), tail), halo)


def analyze_beam(data, charge, *,
                 current_bins='auto',
                 filter_size=1,
                 slice_percent=0.1,
                 slice_with_peak_current=True,
                 min_particles=20):
    """Calculate beam parameters.

    :param Pandas.DataFrame data: particle data.
    :param int/'auto' current_bins: No. of bins to calculate the current
        profile.
    :param int/float filter_size: Standard deviation of the Gaussian kernel
        of the 1D Gaussian filter used for current profile calculation.
    :param float slice_percent: percent of the slice bunch length to the
        total bunch length.
    :param bool slice_with_peak_current: True for calculating slice
        properties of the slice with peak current; False for calculating
        slice properties of the slice in the center of the bunch.
    :param int min_particles: minimum number of particles required for
        phasespace analysis.

    :return BeamParameters: Beam parameters.
    """
    params = BeamParameters()

    n0 = len(data)
    params.n = n0  # Number of particles after processing
    params.q = charge / n0  # charge per particle

    # Too few particles may cause error during the following
    # calculation, e.g. negative value in sqrt.
    if n0 < min_particles:
        raise RuntimeError(f"Too few particles {n0} in the phasespace")

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

    currents, centers = compute_current_profile(
        data['t'], current_bins, params.charge)
    params.I_peak = currents.max()
    params.current_dist = [centers, currents]

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

    filtered_currents = gaussian_filter1d(currents, sigma=filter_size)
    if slice_with_peak_current and params.charge != 0.0:
        Ct_slice = centers[np.argmax(filtered_currents)]  # currents could be all 0
    else:
        Ct_slice = params.Ct

    dt_slice = 4 * params.St * slice_percent  # assume 4-sigma full bunch length
    slice_data = sorted_data[(sorted_data.t > Ct_slice - dt_slice / 2) &
                             (sorted_data.t < Ct_slice + dt_slice / 2)]

    if len(slice_data) < min_particles:
        raise RuntimeError(f"Too few particles {len(slice_data)} in the slice")

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


def analyze_line(data, func, *, min_particles=5):
    """Calculate line parameters.

    :param Pandas.DataFrame data: line data.
    :param callable func: range of the z coordinate.
    :param int min_particles: minimum number of particles required
        for line analysis.

    :return: A LineParameters instance.
    """
    if len(data) < min_particles:
        raise RuntimeError(f"Too few points {len(data)} in the line")

    params = LineParameters()

    params.z = func(data['z'])
    params.gamma = func(data['gamma'])
    params.SdE = func(data['sde'])
    params.Sx = func(data['sx'])
    params.Sy = func(data['sy'])
    params.Sz = func(data['sz'])
    params.betax = func(data['betax'])
    params.betay = func(data['betay'])
    params.alphax = func(data['alphax'])
    params.alphay = func(data['alphay'])
    params.emitx = func(data['emitx'])
    params.emity = func(data['emity'])
    params.emitx_tr = func(data['emitx_tr'])
    params.emity_tr = func(data['emity_tr'])

    return params


def check_data_file(filepath):
    """Check the status of a given file."""
    if not os.path.isfile(filepath):
        raise DataFileNotFoundError(filepath + " does not exist!")
    if not os.path.getsize(filepath):
        raise DataFileEmptyError(filepath + " is empty!")
