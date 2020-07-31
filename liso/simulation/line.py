"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
from abc import abstractmethod, ABC
import math

from ..data_processing import parse_line


class Line(ABC):
    """Line abstract class.

    This class has a method load_data() which reads data from the
    files which record the evolutions of beam parameters.
    """
    def __init__(self, name, rootname, zlim=(-math.inf, math.inf)):
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
        self.zlim = zlim

    @abstractmethod
    def load_data(self):
        """Load data from output files.

        The rootname of these files is self.rootname.
        """
        pass

    def __str__(self):
        text = 'Name: %s\n' % self.name
        text += 'Rootname: %s\n' % self.rootname
        return text


class AstraLine(Line):
    """Line for Astra simulation."""
    def load_data(self):
        """Override the abstract method."""
        return parse_line("a", self.rootname)


class ImpacttLine(Line):
    """Line for Impact-T simulation."""
    def load_data(self):
        """Override the abstract method."""
        return parse_line("t", self.rootname)


class ImpactzLine(Line):
    """Line for Impact-Z simulation."""
    def load_data(self):
        """Override the abstract method."""
        return parse_line("z", self.rootname)


class GenesisLine(Line):
    """Line for Genesis simulation."""
    def load_data(self):
        """Override the abstract method."""
        return parse_line("g", self.rootname)
