"""
Author: Jun Zhu

"""
from abc import abstractmethod

from ..data_processing import parse_astra_line
from ..data_processing import parse_impactt_line
from ..data_processing import parse_impactz_line
from ..data_processing import parse_genesis_line
from ..backend import config
from .stats import Stats


V_LIGHT = config['vLight']
M_E = config['me']
Q_E = config['qe']
INF = config['INF']

CONST_E = M_E*V_LIGHT**2/Q_E


class Line(object):
    """Store the beam evolution and its statistics

    """
    def __init__(self, name, rootname, zlim=(-INF, INF)):
        """Initialize BeamStats object

        :param root_name: string
            The root name of the output files. For Impact-T files,
            root_name will be set to 'fort' if not given.
        :param zlim: tuple

        """
        self.name = name
        self.rootname = rootname

        if not isinstance(zlim, tuple):
            raise TypeError("zlim should be a tuple!")
        if len(zlim) != 2 or zlim[0] >= zlim[1]:
            raise ValueError("zlim should have the form (z_min, z_max)!")
        self.z_min = zlim[0]
        self.z_max = zlim[1]

    @abstractmethod
    def load_data(self):
        """Load data from file."""
        raise NotImplemented

    def get_data(self):
        """Update attributes of attributes"""
        data = self.load_data()

        # Slice data in the range of self.z_lim
        z_max = min(data['z'].max(), self.z_max)
        z_min = max(data['z'].min(), self.z_min)

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

        for key, value in self.__dict__.items():
            if isinstance(value, Stats):
                value.update(data[key])

    def __str__(self):
        text = 'Name: %s\n' % self.name
        text += 'Rootname: %s\n' % self.rootname

        return text


class AstraLine(Line):
    def load_data(self):
        """Load data from file."""
        return parse_astra_line(self.rootname)


class ImpacttLine(Line):
    def load_data(self):
        """Load data from file."""
        return parse_impactt_line(self.rootname)


class ImpactzLine(Line):
    def load_data(self):
        """Load data from file."""
        return parse_impactz_line(self.rootname)


class GenesisLine(Line):
    def load_data(self):
        """Load data from file."""
        return parse_genesis_line(self.rootname)
