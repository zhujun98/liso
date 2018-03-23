from abc import ABC, abstractmethod
from ..config import Config

INF = Config.INF


class PassiveOptimizationElements(ABC):
    """Abstract class for Objective and Constraint."""
    def __init__(self, name, *, expr=None, scale=1.0, func=None):
        """Initialization.

        :param name: string
            Variable Name.
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
        self.name = name
        self.value = INF

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

    @abstractmethod
    def __repr__(self):
        raise NotImplemented

    @abstractmethod
    def __str__(self):
        raise NotImplemented
