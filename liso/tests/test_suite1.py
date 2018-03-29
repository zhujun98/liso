import unittest

from .test_dataAnalysis import TestAnalyzeBeam
from .test_generateInput import TestGenerateInput


def suite():
    suite = unittest.TestSuite()

    suite.addTests(unittest.makeSuite(TestAnalyzeBeam))
    suite.addTests(unittest.makeSuite(TestGenerateInput))

    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
