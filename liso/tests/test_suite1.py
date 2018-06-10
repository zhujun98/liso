"""
Test suite for tests without involving accelerator codes.

Author: Jun Zhu, zhujun981661@gmail.com
"""
import warnings

from .test_dataAnalysis import TestAnalyzeBeam
from .test_generateInput import TestGenerateInput
from .test_alpso import TestALPSO
from .test_nelderMead import TestNelderMead
try:
    from .test_sdpen import TestSDPEN
except ImportError:
    warnings.warn("Skip TestSDPEN due to ImportError!", RuntimeWarning)
