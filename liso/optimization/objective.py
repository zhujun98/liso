#!/usr/bin/env python
"""
Objective class.

Author: Jun Zhu
"""
from ..backend import config

INF = config['INF']


class Objective(object):
    """Optimization Objective Class"""
    def __init__(self, name):
        """Objective Class Initialization

        :param name: string
            Objective name.
        """
        self.name = name
        self.value = INF

    def __repr__(self):
        return '{:^12}  {:^12.4e}\n'.format(self.name[:12], self.value)

    def __str__(self):
        return '{:^12}  {:^12}\n'.format('Name', 'Value') + self.__repr__()
