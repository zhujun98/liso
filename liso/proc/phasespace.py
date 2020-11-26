"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import numpy as np
import pandas as pd

from .beam_parameters import BeamParameters
from .phasespace_analysis import (
    compute_canonical_emit, compute_twiss, compute_current_profile,
    gaussian_filter1d
)
from ..exceptions import LisoRuntimeError


class Phasespace:

    columns = ('x', 'px', 'y', 'py', 'z', 'pz', 't')

    def __init__(self, data, charge):
        """Initialization.

        :param pandas.DataFrame data: phasespace data containing the
            following entries which can be accessed via [] operator:
            x (m), px (mc), y (m), py (mc), z (m), pz (mc), t (s)
        :param None/float charge: bunch charge
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas.DataFrame")

        if set(data.columns) != set(self.columns):
            raise ValueError(f"Data can only have columns: {self.columns}: "
                             f"actual {data.columns}")

        self._data = data
        self.charge = charge

    def __getitem__(self, item):
        try:
            return self._data[item]
        except KeyError:
            data = self._data

            if item == 'dt':
                return data['t'] - data['t'].mean()

            if item == 'p':
                return np.sqrt(data['px'] ** 2 + data['py'] ** 2 + data['pz'] ** 2)

            if item == 'xp':
                return data['px'] / data['pz']

            if item == 'yp':
                return data['py'] / data['pz']

            if item == 'dz':
                z_ave = data['z'].mean()
                return data['z'] - z_ave

            if item == 'delta':
                p = np.sqrt(data['px'] ** 2 + data['py'] ** 2 + data['pz'] ** 2)
                p_ave = p.mean()
                return 100. * (p - p_ave) / p

            raise

    def __len__(self):
        return self._data.__len__()

    def reindex(self, *args, **kwargs):
        self._data.reindex(*args, **kwargs)

    def slice(self, start=None, stop=None, step=None, *, inplace=False):
        """Slice the phasespace.

        :param None/int start: starting integer when the slicing of the
            phasespace starts.
        :param None/int stop: integer until which the slicing takes place.
            The slicing stops at index stop - 1.
        :param None/int step: integer value which determines the increment
            between each index for slicing.
        :param bool inplace: True for inplace operation.

        :return Phasespace: the sliced phasespace instance.
        """
        slicer = slice(start, stop, step)
        sliced = self._data[slicer]

        if inplace:
            self.charge *= len(sliced) / len(self._data)
            self._data = sliced
            return self

        return Phasespace(sliced, self.charge * len(sliced) / len(self._data))

    def cut_halo(self, percent):
        """Remove halo from the phasespace.

        :param float percent: percentage of particles to be removed based on
            their transverse distance to the bunch centroid.
        """
        if 1.0 > percent > 0.0:
            data = self._data
            n = int(len(data) * (1 - percent))
            data['r'] = np.sqrt(data['x'] ** 2 + data['y'] ** 2)
            data = data.reindex(data['r'].sort_values(ascending=True).index)
            del data['r']
            self.slice(None, n, inplace=True)

    def cut_tail(self, percent):
        """Remove tail from the phasespace.

        :param float percent: percentage of particles to be removed
            in the tail.
        """
        if 1.0 > percent > 0.0:
            data = self._data
            n = int(len(data) * (1 - percent))
            # User median() to deal with extreme outliers
            data['t_'] = data['t'] - data['t'].median()
            data = data.reindex(
                data['t_'].abs().sort_values(ascending=True).index)
            del data['t_']
            self.slice(None, n, inplace=True)

    def rotate(self, angle):
        """Rotate the phasespace.

        :param float angle: angle of the rotation in rad.
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

            data = self._data
            pos = [data['x'], data['y'], data['z']]
            cm_pos = [np.mean(data['x']), np.mean(data['y']), np.mean(data['z'])]
            mom = [data['px'], data['py'], data['pz']]
            cm_mom = [np.mean(data['px']), np.mean(data['py']),
                      np.mean(data['pz'])]

            [data['x'], data['y'], data['z']] = transformation(pos, cm_pos)
            [data['px'], data['py'], data['pz']] = transformation(mom, cm_mom)

    def analyze(self, *,
                current_bins='auto',
                filter_size=1,
                slice_percent=0.1,
                slice_with_peak_current=True,
                min_particles=20):
        """Calculate beam parameters.

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
        charge = 0 if self.charge is None else self.charge

        data = self._data
        params = BeamParameters()

        n0 = len(data)
        params.n = n0  # Number of particles after processing
        params.q = charge / n0  # charge per particle

        # Too few particles may cause error during the following
        # calculation, e.g. negative value in sqrt.
        if n0 < min_particles:
            raise LisoRuntimeError(f"Too few particles {n0} in the phasespace")

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
        sorted_data = data.reindex(
            data['t'].abs().sort_values(ascending=True).index)

        try:
            filtered_currents = gaussian_filter1d(currents, sigma=filter_size)
            if slice_with_peak_current and params.charge != 0.0:
                # currents could be all 0
                Ct_slice = centers[np.argmax(filtered_currents)]
            else:
                Ct_slice = params.Ct

            # assume 4-sigma full bunch length
            dt_slice = 4 * params.St * slice_percent
            slice_data = sorted_data[(sorted_data.t > Ct_slice - dt_slice / 2) &
                                     (sorted_data.t < Ct_slice + dt_slice / 2)]

            if len(slice_data) < min_particles:
                raise LisoRuntimeError(
                    f"Too few particles {len(slice_data)} in the slice")

            p_slice = np.sqrt(slice_data['pz'] ** 2
                              + slice_data['px'] ** 2
                              + slice_data['py'] ** 2)

            params.emitx_slice = compute_canonical_emit(
                slice_data.x, slice_data.px)
            params.emity_slice = compute_canonical_emit(
                slice_data.y, slice_data.py)
            params.Sdelta_slice = p_slice.std(ddof=0) / p_slice.mean()
            params.dt_slice = slice_data.t.max() - slice_data.t.min()

            # The output will be different from the output by msddsplot
            # because the slightly different No of particles sliced. It
            # affects the correlation calculation since the value is
            # already very close to 1.
            params.Sdelta_un = \
                params.Sdelta_slice \
                * np.sqrt(1 - (slice_data['t'].corr(p_slice)) ** 2)

        except Exception:
            pass

        return params

    @classmethod
    def from_columns(cls,
                     x=None, px=None,
                     y=None, py=None,
                     z=None, pz=None, t=None, *, charge=0.):
        df = pd.DataFrame({
            'x': x, 'px': px, 'y': y, 'py': py, 'z': z, 'pz': pz, 't': t
        })
        return cls(df, charge)

    @classmethod
    def from_dict(cls, data, charge=0.):
        return cls(pd.DataFrame(data), charge)
