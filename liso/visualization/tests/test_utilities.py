import unittest


from liso.visualization.vis_utils import (
    get_default_unit, get_label, get_html_label, get_line_column_by_name,
    get_unit_label_and_scale
)


class TestUtilities(unittest.TestCase):

    def testGetlabel(self):
        self.assertEqual(get_label('gamma'), get_label('Gamma'))

    def testGetHtmlLabel(self):
        self.assertEqual(get_html_label('sx'), get_html_label('sX'))

    def testGetDefaultUnit(self):
        self.assertEqual('mm', get_default_unit('x'))
        self.assertEqual('m', get_default_unit('z'))
        self.assertEqual('um', get_default_unit('emitx'))

    def testGetUnitLabelAndScale(self):
        self.assertTupleEqual(('(kA)', 1.e-3), get_unit_label_and_scale('kA'))
        self.assertTupleEqual(('(MeV)', 1.e-6), get_unit_label_and_scale('mev'))

    def testGetLineColumnByName(self):
        pass
