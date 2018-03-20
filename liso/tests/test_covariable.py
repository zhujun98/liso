"""
Unittest for InputGenerator
"""
import unittest

from ..optimization.covariable import Covariable


class TestCovariable(unittest.TestCase):
    def test_initialization(self):
        self.assertRaises(TypeError, Covariable, 'a', 'b', 'c')
        self.assertRaises(ValueError, Covariable, 'a', 'b', [1, 2])
        self.assertRaises(ValueError, Covariable, 'a', ['b', 'c'], [1])
        self.assertRaises(TypeError, Covariable, 'a', ['b', 'c'], [1, 1], [0, 0])
        self.assertRaises(TypeError, Covariable, 'a', 1, 1, 0)
        Covariable('a', 'b', 1)
        Covariable('a', ['b', 'c', 'd'], [1, 1, 1], 0)


if __name__ == "__main__":
    unittest.main()
