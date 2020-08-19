import os
import os.path as osp
import pathlib
import tempfile
import unittest

from liso.io import TempSimulationDirectory


class TestTempDir(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = "/tmp/tmp_simulation_dir"

    def tearDown(self):
        # test garbage collected
        self.assertFalse(osp.isdir(self._tmp_dir))

    def testGeneral(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(FileExistsError):
                TempSimulationDirectory(tmp_dir)
            self.assertTrue(osp.isdir(tmp_dir))

        tmp_dir = self._tmp_dir
        with TempSimulationDirectory(tmp_dir) as swd:
            self.assertEqual(tmp_dir, swd)
            self.assertTrue(osp.isdir(tmp_dir))
            pathlib.Path(osp.join(tmp_dir, "tmp_file")).touch()
            os.mkdir(osp.join(tmp_dir, "tmp_folder"))
        self.assertFalse(osp.isdir(tmp_dir))

        t = TempSimulationDirectory(tmp_dir)
        self.assertTrue(osp.isdir(tmp_dir))
