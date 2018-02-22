#!/usr/bin/python
"""
Author: Jun Zhu
"""
import copy
from numbers import Number


class Covariable(object):
    """Covariable class.

    Covariable is a variable which changes along with one or several
    variables.
    """
    def __init__(self, name, vars, a=1.0, b=0.0):
        """Initialize CoVariable object

        The value of the variable is calculated by:
        covar = [a].*vars + b

        :param name: string
            Name of the co-variable.
        :param vars: string/list
            Name(s) of the dependent variable(s).
        :param a: float/list
            Coefficient(s).
        :param b: float
            Coefficient.
        """
        self.name = name

        if isinstance(vars, str) and isinstance(a, Number):
            self.vars = [vars]
            self.a = [a]
        elif hasattr(vars, '__iter__') and (not isinstance(a, str) and hasattr(a, '__iter__')):
            if len(vars) != len(a):
                raise ValueError("vars and a must have the same length!\n")
            self.vars = copy.copy(vars)
            self.a = copy.copy(a)
        else:
            raise TypeError("Invalid input!")

        if not isinstance(b, Number):
            raise TypeError("b must be a number!")
        self.b = b

        if name in self.vars:
            raise ValueError("Covariable and one dependent variable are the same.\n")

        self.value = None

    def __str__(self):
        """Print structured list of co-variables.

        Overwrite the original method.
        """
        text = '  {:18}  {:16}  {:11}  {:11}\n'.format('Name', 'Dependents', 'a', 'b')

        for i in range(len(self.vars)):
            if i == 0:
                text += '  {:18}  {:16}  {:11.4e}  {:11.4e}\n'.format(
                    self.name, self.vars[i], self.a[i], self.b)
            else:
                text += '  {:18}  {:16}  {:11.4e}\n'.format(
                    '', self.vars[i], self.a[i])

        return text
