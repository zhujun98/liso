"""
Unittest for InputGenerator

Author: Jun Zhu, zhujun981661@gmail.com
"""
import os
import unittest

from liso.simulation.simulation_utils import generate_input


test_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'files4test'
))
input_file = os.path.join(test_path, "injector.in")


class TestGenerateInput(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(test_path, "injector.in.000")) as fp:
            self.template = tuple(fp.readlines())

    def test_raises(self):
        mapping = {'gun_gradient': 10, 'gun_phase0': 20}
        self.assertRaises(KeyError, generate_input, self.template, mapping,
                          input_file)

    def test_not_raise(self):
        mapping = {'gun_gradient': 10, 'gun_phase': 20, 'tws_gradient': 30}
        generate_input(self.template, mapping, input_file)

        mapping = {'gun_gradient': 10, 'gun_phase': 20}
        generate_input(self.template, mapping, input_file)


if __name__ == "__main__":
    unittest.main()
