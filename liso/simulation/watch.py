"""
Note: In the calculation (except the canonical emittance), the
particles are drifted back to the center of the bunch without
considering the collective effects!!!

Author: Jun Zhu

"""
from abc import abstractmethod

import numpy as np

from ..data_processing import parse_astra_phasespace
from ..data_processing import parse_impactt_phasespace
from ..data_processing import parse_impactz_phasespace
from ..data_processing import parse_genesis_phasespace
from ..data_processing import compute_canonical_emit
from ..data_processing import compute_twiss
from ..data_processing import gaussian_filter1d
from .beam_parameters import BeamParameters
from ..backend import config


V_LIGHT = config['vLight']
M_E = config['me']
Q_E = config['qe']
INF = config['INF']

CONST_E = M_E*V_LIGHT**2/Q_E


class Watch(object):
    """Beam parameters for a phasespace."""
    def __init__(self, name, pfile, *,
                 slice_percent=0.1,
                 cut_halo=0.0,
                 cut_tail=0.0,
                 current_bins='auto',
                 filter_size=1,
                 min_pars=5,
                 slice_with_peak_current=True):
        """
        pfile: string
            Path name of the particle file.
        slice_percent: float
            Percent of the slice bunch length to the total bunch length.
        cut_halo: None/float
            Percentage of particles to be removed based on their
            transverse distance to the bunch centroid. Applied
            before tail cutting.
        cut_tail: None/float
            Percentage of particles to be removed in the tail.
        current_bins: int/'auto'
            No. of bins to calculate the current profile.
        filter_size: int/float
            Standard deviation of the Gaussian kernel of the 1D Gaussian
            filter used for current profile calculation.
        min_pars: int
            Minimum allowed number of particles in the data.
        slice_with_peak_current: Boolean
            True for calculating slice properties of the slice with peak
            current; False for calculating slice properties of the slice
            in the center of the bunch.
        """
        self.name = name
        self.pfile = pfile
        self.charge = None

        self._slice_percent = 1.0  # property
        self.slice_percent = slice_percent
        self._cut_halo = 0.0  # property
        self.cut_halo = cut_halo
        self._cut_tail = 0.0  # property
        self.cut_tail = cut_tail

        self.current_bins = current_bins
        self.filter_size = filter_size
        self.slice_with_peak_current = slice_with_peak_current

        self._min_pars = min_pars

    @property
    def slice_percent(self):
        return self._slice_percent

    @slice_percent.setter
    def slice_percent(self, value):
        if type(value) in (int, float) and 0.0 < value <= 1.0:
            self._slice_percent = value
        else:
            raise ValueError("Invalid input for slice_percent: {}".format(value))

    @property
    def cut_halo(self):
        return self._cut_halo

    @cut_halo.setter
    def cut_halo(self, value):
        if type(value) in (int, float) and 0.0 <= value < 1.0:
            self._cut_halo = value
        else:
            raise ValueError("Invalid input for cut_halo: {}".format(value))

    @property
    def cut_tail(self):
        return self._cut_tail

    @cut_tail.setter
    def cut_tail(self, value):
        if type(value) in (int, float) and 0.0 <= value < 1.0:
            self._cut_tail = value
        else:
            raise ValueError("Invalid input for cut_tail: {}".format(value))

    @abstractmethod
    def _load_data(self):
        pass

    def get_data(self):
        """"""
        data = self._load_data()

        params = BeamParameters()

        n0 = len(data)
        params.n = n0  # Number of particles after processing
        params.q = self.charge / n0  # charge per particle

        # Cut the halo of the bunch
        params.n = int(params.n*(1 - self.cut_halo))

        # TODO: avoid creating a column and then deleting it
        data['r'] = np.sqrt(data['x'] ** 2 + data['y'] ** 2)
        data = data.reindex(data['r'].sort_values(ascending=True).index)
        data = data[:params.n]
        del data['r']

        # Cut the tail of the bunch
        params.n = int(params.n*(1 - self.cut_tail))
        data['t'] -= data['t'].median()  # User median() to deal with extreme outliers
        data = data.reindex(data['t'].abs().sort_values(ascending=True).index)
        data = data[:params.n]
        data['t'] -= data['t'].mean()

        # Too few particles may cause error during the following
        # calculation, e.g. negative value in sqrt.
        if params.n < self._min_pars:
            raise ValueError("Too few particles ({}) in the data!".format(params.n))

        p = np.sqrt(data['pz']**2 + data['px']**2 + data['py']**2)

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
        counts, edges = np.histogram(data['t'], bins=self.current_bins)
        step_size = edges[1] - edges[0]
        centers = edges[:-1] + step_size / 2
        current = counts / float(len(data)) * params.charge / step_size
        current = gaussian_filter1d(current, sigma=self.filter_size)
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

        if self.slice_with_peak_current is True:
            Ct_slice = centers[np.argmax(current)]
        else:
            Ct_slice = params.Ct

        dt_slice = 4*params.St*self.slice_percent  # assume 4-sigma full bunch length
        slice_data = sorted_data[(sorted_data.t > Ct_slice - dt_slice/2) &
                                 (sorted_data.t < Ct_slice + dt_slice/2)]

        if len(slice_data) < self._min_pars:
            raise ValueError("Too few particles ({}) in the slice data!".
                             format(len(slice_data)))

        p_slice = np.sqrt(slice_data['pz']**2 + slice_data['px']**2 + slice_data['py']**2)

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

    def __str__(self):
        text = 'Name: %s\n' % self.name
        text += 'Particle file: %s\n' % self.pfile

        return text


class AstraWatch(Watch):
    """Watch for Astra simulation."""
    def __init__(self, name, pfile, **kwargs):
        """"""
        super().__init__(name, pfile, **kwargs)

    def _load_data(self):
        """Read data from file."""
        data, self.charge = parse_astra_phasespace(self.pfile)
        return data


class ImpacttWatch(Watch):
    """Watch for ImpactT simulation."""
    def __init__(self, name, pfile, charge, **kwargs):
        """"""
        super().__init__(name, pfile, **kwargs)
        self.charge = charge

    def _load_data(self):
        """Read data from file."""
        data = parse_impactt_phasespace(self.pfile)
        return data


class ImpactzWatch(Watch):
    """Watch for ImpactZ simulation."""
    def __init__(self, name, pfile, charge, **kwargs):
        """"""
        super().__init__(name, pfile, **kwargs)
        self.charge = charge

    def _load_data(self):
        """Read data from file."""
        data = parse_impactz_phasespace(self.pfile)
        return data


class GenesisWatch(Watch):
    """Watch for Genesis simulation."""
    def __init__(self, name, pfile, charge, **kwargs):
        """"""
        super().__init__(name, pfile, **kwargs)
        self.charge = charge

    def _load_data(self):
        """Read data from file."""
        data = parse_genesis_phasespace(self.pfile)
        return data
