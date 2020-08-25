"""
Author: Jun Zhu

"""


class SimulationError(Exception):
    """Raised when a simulation fails."""
    pass


class SimulationSuccessiveFailureError(SimulationError):
    """Inherited from SimulationError.

    Raised No. of successive failures exceeds the maximum allowed
    number.
    """
    pass


class BeamAnalysisError(Exception):
    """Raised when there is error in beam analysis."""
    pass


class BeamParametersInconsistentError(BeamAnalysisError):
    """Inherited from BeamAnalysisError.

    Raised when there is inconsistency in beam parameters.
    """
    pass


class OptimizationError(Exception):
    """Raise if there is error in optimization."""
    pass


class OptimizationConstraintSupportError(OptimizationError):
    """Raise if an optimizer does not support certain constraint."""
    pass
