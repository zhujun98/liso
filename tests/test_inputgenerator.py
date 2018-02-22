"""
Unittest for InputGenerator
"""
import unittest

from liso.simulation.beamline import AstraBeamline, ImpacttBeamline
from liso.simulation.smlt_utils import generate_input


TEMPLATE_FILE1 = "../examples/astra_basic/injector.in.000"
TEMPLATE_FILE2 = "../examples/impactt_basic/ImpactT.in.000"
INPUT_FILE1 = "../examples/astra_basic/injector.in"
INPUT_FILE2 = "../examples/impactt_basic/ImpactT.in"


class TestGenerateInput(unittest.TestCase):
    def setUp(self):
        self.b1 = AstraBeamline('astra', INPUT_FILE1, template=TEMPLATE_FILE1)
        self.b2 = ImpacttBeamline('impact', INPUT_FILE2, template=TEMPLATE_FILE2, charge=0)

    def test_raises(self):
        mapping = 'laser'
        self.assertRaises(TypeError, generate_input, self.b1, mapping)

        mapping = {'laser_spot': 10, 'main_sole_b': 20}
        self.assertRaises(TypeError, generate_input, 1, mapping)

        mapping = {'n_col': 2, 'n_row': 1, 'MQZM1_G': 0.5}
        self.assertRaises(KeyError, generate_input, self.b2, mapping)

    def test_astra_input(self):
        mapping = {'laser_spot': 10, 'main_sole_b': 20}
        generate_input(self.b1, mapping)
        # the second one should not raise
        generate_input(self.b1, mapping)

    def test_impact_input(self):
        mapping = {'n_col': 2, 'n_row': 1, 'MQZM1_G': 0.5, 'MQZM2_G': -0.5}
        generate_input(self.b2, mapping)


if __name__ == "__main__":
    unittest.main()
