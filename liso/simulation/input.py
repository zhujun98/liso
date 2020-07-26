"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
from abc import abstractmethod


class InputGenerator(object):
    def __init__(self):
        """Initialization."""

    @abstractmethod
    def add_quad(self):
        raise NotImplemented

    @abstractmethod
    def add_dipole(self):
        raise NotImplemented

    @abstractmethod
    def add_tws(self):
        raise NotImplemented

    @abstractmethod
    def add_gun(self):
        raise NotImplemented


class AstraInputGenerator(InputGenerator):
    pass


class ImpacttInputGenerator(InputGenerator):
    pass
