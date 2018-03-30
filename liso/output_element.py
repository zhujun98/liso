"""
Author: Jun Zhu
"""

from .operation_element import OperationElement
from .config import Config

INF = Config.INF


class OutputElement(OperationElement):
    """Abstract class a descriptive parameter.

    The value of this parameter can be either calculated by parsing
    a given string or evaluating a given function.
    """
    def __init__(self, name, *, expr=None, scale=1.0, func=None):
        """Initialization.

        :param expr: string
            Expression for an attribute of a BeamParameters instance
            or a LineParameters instance,
            e.g. gun.out.Sx, chicane.max.betax.

            Ignored if 'func' is defined.
        :param scale: float
            Multiplier of the value evaluated from 'expr'.
        :param func: functor
            Used fo update the constraint value.
        """
        super().__init__(name)

        if expr is None and func is None:
            raise ValueError("Unknown expression!")

        self.expr = None
        self.scale = scale
        self.func = None
        if func is None:
            if expr is not None and not isinstance(expr, str):
                raise TypeError("'expr' must be a string!")

            self.expr = expr.split(".")
            if len(self.expr) != 3:
                raise ValueError("'expr' must have the form "
                                 "'beamline_name.WatchParameters_name.param_name' "
                                 "or "
                                 "'beamline_name.LineParameters_name.param_name'")
        else:
            if not hasattr(func, '__call__'):
                raise TypeError("'func' must be callable!")
            self.func = func
