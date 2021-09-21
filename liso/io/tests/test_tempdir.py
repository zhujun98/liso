from pathlib import Path
import tempfile
import unittest

from liso.io import TempSimulationDirectory


_ROOT_DIR = Path(__file__).parent


class TestTempDir(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = _ROOT_DIR.joinpath("temp_dir_test")

    def tearDown(self):
        # test garbage collected
        assert not self._tmp_dir.is_dir()

    def testDirAlreadyExists(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(FileExistsError):
                TempSimulationDirectory(tmp_dir)

    def testDirWithFile(self):
        tmp_dir = self._tmp_dir
        with TempSimulationDirectory(tmp_dir) as swd:
            assert tmp_dir == swd
            assert tmp_dir.is_dir()
            Path(tmp_dir, "tmp_file").touch()
            path_dir = Path(tmp_dir, "tmp_folder")
            path_dir.mkdir()
        assert not path_dir.is_dir()

        _ = TempSimulationDirectory(tmp_dir)
        assert tmp_dir.is_dir()
