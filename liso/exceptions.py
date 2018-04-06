"""
Author: Jun Zhu

"""


class SimulationError(Exception):
    """Raised when a simulation fails."""
    pass


class SimulationNotFinishedProperlyError(SimulationError):
    """Inherited from SimulationError.

    Raised when there is error in Beamline.simulate().
    """
    pass


class SimulationSuccessiveFailureError(SimulationError):
    """Inherited from SimulationError.

    Raised No. of successive failures exceeds the maximum allowed
    number."""
    pass


class BeamAnalysisError(Exception):
    """Raised when there is error in beam analysis."""
    pass


class BeamParametersInconsistentError(BeamAnalysisError):
    """Inherited from BeamAnalysisError.

    Raised when there is inconsistency in beam parameters.
    """
    pass


class TooFewOutputParticlesError(BeamAnalysisError):
    """Inherited from BeamAnalysisError.

    Raised when there are not enough particles in the output.
    """
    pass


class WatchUpdateFailError(Exception):
    """Raised if watch update fails.

    e.g. file is missing or data format is wrong
    """
    pass


class OutUpdateFailError(WatchUpdateFailError):
    """Inherited from WatchUpdateFailError

    Raised if 'out' update fails."""
    pass


class LineUpdateFailError(Exception):
    """Raise if line update fails.

    e.g file is missing or data format is wrong
    """
    pass


class OptimizationError(Exception):
    """Raise if there is error in optimization."""
    pass


class OptimizationConstraintSupportError(OptimizationError):
    """Raise if an optimizer does not support certain constraint."""
    pass
