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


class BeamlineInputFileNotFoundError(FileNotFoundError):
    """Inherited from FileNotFoundError.

    Raised if the simulation input file does not exist.
    """
    pass


class WatchFileNotFoundError(FileNotFoundError):
    """Inherited from FileNotFoundError.

    Raised if Watch.pfile does not exist.
    """
    pass


class LineFileNotFoundError(FileNotFoundError):
    """Inherited from FileNotFoundError.

    Raised if any one of the Line.rootname + '.suffix' does not exit.
    """
    pass
