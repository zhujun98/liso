"""
Unittest for InputGenerator
"""
import unittest

from ..simulation.simulation_utils import generate_input

TEMPLATE = "liso/tests/jitter_test/injector.in.000"
OUTPUT = "liso/tests/astra_gun/injector.in"


class TestGenerateInput(unittest.TestCase):
    def setUp(self):
        with open(TEMPLATE) as fp:
            self.template = tuple(fp.readlines())

    def test_raises(self):
        mapping = {'gun_gradient': 10, 'gun_phase0': 20}
        self.assertRaises(KeyError, generate_input, self.template, mapping, OUTPUT)

    def test_not_raise(self):
        mapping = {'gun_gradient': 10, 'gun_phase': 20, 'tws_gradient': 30}
        generate_input(self.template, mapping, OUTPUT)

        mapping = {'gun_gradient': 10, 'gun_phase': 20}
        generate_input(self.template, mapping, OUTPUT)


if __name__ == "__main__":
    unittest.main()
