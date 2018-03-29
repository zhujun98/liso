import unittest

from .test_jitter import TestJitter
from .test_global_optimizer import TestGlobalOptimizer
from .test_multi_code_optimization import TestMultiCodeOptimization


def suite():
    suite = unittest.TestSuite()

    suite.addTests(unittest.makeSuite(TestJitter))
    suite.addTests(unittest.makeSuite(TestGlobalOptimizer))
    suite.addTests(unittest.makeSuite(TestMultiCodeOptimization))

    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
