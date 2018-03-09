"""
Note: In the calculation (except the canonical emittance), the
particles are drifted back to the center of the bunch without
considering the collective effects!!!

Author: Jun Zhu

"""
from abc import abstractmethod
from abc import ABC

from ..data_processing import parse_astra_phasespace
from ..data_processing import parse_impactt_phasespace
from ..data_processing import parse_impactz_phasespace
from ..data_processing import parse_genesis_phasespace
from ..backend import config


V_LIGHT = config['vLight']
M_E = config['me']
Q_E = config['qe']
INF = config['INF']

CONST_E = M_E*V_LIGHT**2/Q_E


class Watch(ABC):
    """Watch abstract class.

    The class has a method get_data() which returns a BeamParameter
    object.
    """
    def __init__(self, name, pfile, *,
                 cut_halo=0.0,
                 cut_tail=0.0,
                 current_bins='auto',
                 filter_size=1,
                 slice_percent=0.1,
                 slice_with_peak_current=True):
        """Initialization.

        :param name: string
            Name of the Watch object.
        :param pfile: string
            Path name of the particle file.
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
        """
        self.name = name
        self.pfile = pfile

        self._slice_percent = 1.0  # property
        self.slice_percent = slice_percent
        self._cut_halo = 0.0  # property
        self.cut_halo = cut_halo
        self._cut_tail = 0.0  # property
        self.cut_tail = cut_tail

        self.current_bins = current_bins
        self.filter_size = filter_size
        self.slice_with_peak_current = slice_with_peak_current

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
    def load_data(self):
        """Read data from the particle file.

        The particle file is self.pfile.
        """
        pass

    def __str__(self):
        text = 'Name: %s\n' % self.name
        text += 'Particle file: %s\n' % self.pfile
        return text


class AstraWatch(Watch):
    """Watch for Astra simulation."""
    def load_data(self):
        """Override the abstract method."""
        data, charge = parse_astra_phasespace(self.pfile)
        return data, charge


class ImpacttWatch(Watch):
    """Watch for Impact-T simulation."""
    def load_data(self):
        """Override the abstract method."""
        data = parse_impactt_phasespace(self.pfile)
        return data, None


class ImpactzWatch(Watch):
    """Watch for Impact-Z simulation."""
    def load_data(self):
        """Override the abstract method."""
        data = parse_impactz_phasespace(self.pfile)
        return data, None


class GenesisWatch(Watch):
    """Watch for Genesis simulation."""
    def load_data(self):
        """Override the abstract method."""
        data = parse_genesis_phasespace(self.pfile)
        return data, None
