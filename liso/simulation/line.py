"""
Author: Jun Zhu

"""
from abc import abstractmethod
from abc import ABC

from .line_parameters import Stats
from .line_parameters import LineParameters
from ..data_processing import parse_astra_line
from ..data_processing import parse_impactt_line
from ..data_processing import parse_impactz_line
from ..data_processing import parse_genesis_line
from ..backend import config


V_LIGHT = config['vLight']
M_E = config['me']
Q_E = config['qe']
INF = config['INF']

CONST_E = M_E*V_LIGHT**2/Q_E


class Line(ABC):
    """Line abstract class.

    The class has a method get_data() which returns a LineParameter
    object.
    """
    def __init__(self, name, rootname, zlim=(-INF, INF)):
        """Initialize BeamStats object

        :param name: string
            Name of the Line object.
        :param rootname: string
            The rootname (including path) of output files.
        :param zlim: tuple, (z_min, z_max)
            Range of the z coordinate.
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
    def _load_data(self):
        """Load data from output files.

        The rootname of these files is self.rootname.
        """
        raise NotImplemented

    def get_data(self):
        """Read data from files and analyse the data.

        :return: A LineParameter object.
        """
        data = self._load_data()

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

        params = LineParameters()
        for key, value in params.__dict__.items():
            if isinstance(value, Stats):
                value.update(data[key])

        return params

    def __str__(self):
        text = 'Name: %s\n' % self.name
        text += 'Rootname: %s\n' % self.rootname
        return text


class AstraLine(Line):
    def _load_data(self):
        """Override the abstract method."""
        return parse_astra_line(self.rootname)


class ImpacttLine(Line):
    def _load_data(self):
        """Override the abstract method."""
        return parse_impactt_line(self.rootname)


class ImpactzLine(Line):
    def _load_data(self):
        """Override the abstract method."""
        return parse_impactz_line(self.rootname)


class GenesisLine(Line):
    def _load_data(self):
        """Override the abstract method."""
        return parse_genesis_line(self.rootname)
