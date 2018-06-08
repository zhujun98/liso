import unittest
import warnings

from .test_dataAnalysis import TestAnalyzeBeam
from .test_generateInput import TestGenerateInput
from .test_alpso import TestALPSO
from .test_nelderMead import TestNelderMead
try:
    from .test_sdpen import TestSDPEN
except ImportError:
    warnings.warn("Skip TestSDPEN due to ImportError!", RuntimeWarning)


def suite():
    suite = unittest.TestSuite()

    suite.addTests(unittest.makeSuite(TestAnalyzeBeam))
    suite.addTests(unittest.makeSuite(TestGenerateInput))
    suite.addTests(unittest.makeSuite(TestALPSO))
    suite.addTests(unittest.makeSuite(TestNelderMead))
    suite.addTests(unittest.makeSuite(TestSDPEN))

    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())