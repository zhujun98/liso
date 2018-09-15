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
    number.
    """
    pass


class CommandNotFoundError(SimulationError):
    """Inherited from SimulationError.

    Raised if a bash command is not found.
    """
    pass


class InputFileNotFoundError(SimulationError):
    """Inherited from SimulationError.

    Raised if the simulation input file is not found.
    """
    pass


class InputFileEmptyError(SimulationError):
    """Inherited from SimulationError.

    Raised if the simulation input file is empty.
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


class DataFileNotFoundError(BeamAnalysisError):
    """Inherited from BeamAnalysisError.

    Raised if the data file is not found.
    """
    pass


class DataFileEmptyError(BeamAnalysisError):
    """Inherited from BeamAnalysisError.

    Raised if the data file is empty.
    """
    pass


class WatchUpdateError(Exception):
    """Raised if watch update fails.

    e.g. file is missing or data format is wrong
    """
    pass


class LineUpdateError(Exception):
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
