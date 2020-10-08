import unittest

from liso.io.file_access import _init_file_open_registry


class TestFileAccess(unittest.TestCase):
    def testGeneral(self):
        _init_file_open_registry()
