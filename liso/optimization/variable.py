"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import math

from ..elements import OperationalElement


class Variable(OperationalElement):
    """Variable Class. Inherited from OperationalElement."""
    def __init__(self, name, type_='c', *, value=0.0, **kwargs):
        """Variable Class Initialization

        :param type_: string
            Variable type
            ('c' - continuous, 'i' - integer, 'd' - discrete), default = 'c'
        :param value: int / float
            Variable Value, default = 0.0
        :param lb: int / float
            Lower boundary.
        :param ub: int / float
            Upper boundary.
        """
        super().__init__(name)

        self.type_ = type_[0].lower()
        if self.type_ == 'c':
            self.value = float(value)
            self.lb = -math.inf
            self.ub = math.inf
            for key in kwargs.keys():
                if key.lower() == 'lb':
                    self.lb = float(kwargs[key])
                if key.lower() == 'ub':
                    self.ub = float(kwargs[key])

        elif type_[0].lower() == 'i':
            self.value = int(value)
            self.lb = None
            self.ub = None
            for key in kwargs.keys():
                if key.lower() == 'lb':
                    self.lb = int(kwargs[key])
                if key.lower() == 'ub':
                    self.ub = int(kwargs[key])

            if self.lb is None or self.ub is None:
                raise ValueError('An integer variable requires both '
                                 'lower and upper boundaries')

        elif type_[0].lower() == 'd':
            self.choices = None
            for key in kwargs.keys():
                if key.lower() == 'choices':
                    self.choices = kwargs[key]
                    break

            if self.choices is None:
                raise ValueError('A discrete variable requires to '
                                 'input an array of choices')

            if not isinstance(value, int):
                raise TypeError("A discrete variable requires the 'value' "
                                "to be a valid index the choices array")
            else:
                self.value = self.choices[int(value)]

            self.lb = 0
            self.ub = len(self.choices) - 1
        else:
            raise ValueError('Unknown variable types: should be '
                             'either c(ontinuous), i(nteger) or d(iscrete)')

        if self.ub < self.lb:
            raise ValueError("Upper bound is smaller than lower bound!")

    def list_item(self):
        if self.type_ == 'd':
            return '{:12}  {:^6}  {:^12.4e}  {:^12.4e}  {:^12.4e}\n'.format(
                self.name[:12], self.type_, self.choices[int(self.value)],
                min(self.choices), max(self.choices))

        return '{:12}  {:^6}  {:^12.4e}  {:^12.4e}  {:^12.4e}\n'.format(
            self.name[:12], self.type_, self.value, self.lb, self.ub)

    def __str__(self):
        return '{:12}  {:^6}  {:^12}  {:^12}  {:^12}\n'.format(
               'Name', 'Type', 'Value', 'Lower Bound', 'Upper Bound') + \
               self.list_item()
