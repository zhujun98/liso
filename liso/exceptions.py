"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""


class SimulationError(Exception):
    """Raised when a simulation fails."""


class SimulationSuccessiveFailureError(SimulationError):
    """Inherited from SimulationError.

    Raised No. of successive failures exceeds the maximum allowed
    number.
    """


class BeamAnalysisError(Exception):
    """Raised when there is error in beam analysis."""


class BeamParametersInconsistentError(BeamAnalysisError):
    """Inherited from BeamAnalysisError.

    Raised when there is inconsistency in beam parameters.
    """


class OptimizationError(Exception):
    """Raise if there is error in optimization."""


class OptimizationConstraintSupportError(OptimizationError):
    """Raise if an optimizer does not support certain constraint."""


# FIXME: why inherited from RuntimeError?
class LisoRuntimeError(RuntimeError):
    """RuntimeError for LISO."""
