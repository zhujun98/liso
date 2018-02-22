"""
Author: Jun Zhu

"""

class SimulationError(Exception):
    """Exceptions raised within the 'simulation' package."""
    pass


class SimulationNotFinishedProperlyError(SimulationError):
    """Raised when there is error in Beamline.simulate()"""
    pass


class BeamlineInputFileNotGeneratedError(SimulationError):
    """"""
    pass


class SimulationSuccessiveFailureError(SimulationError):
    """"""
    pass


class BeamParametersInconsistentError(Exception):
    """"""
    pass


class BeamlineMonitorError(Exception):
    """"""
    pass