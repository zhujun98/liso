import unittest
from unittest.mock import patch
import tempfile
import pathlib

import numpy as np

from liso import EuXFELInterface, MachineScan, open_run
from liso import doocs_channels as dc
from liso.experiment import doocs_channels
from liso.experiment.doocs_interface import DoocsException
from liso.io import ExpWriter
from liso.logging import logger
logger.setLevel('ERROR')

from ...experiment.tests import DoocsDataGenerator as ddgen


_INITIAL_PID = 1000


def _side_effect_read(dataset, address):
    data = dataset[address]
    if data['macropulse'] >= _INITIAL_PID:
        data['data'] = data['data'] + 1  # do not mutate
        data['macropulse'] += 1
    return data


class TestMachineScan(unittest.TestCase):
    def setUp(self):
        self._orig_image_chunk = ExpWriter._IMAGE_CHUNK
        ExpWriter._IMAGE_CHUNK = (3, 2)

    def tearDown(self):
        ExpWriter._IMAGE_CHUNK = self._orig_image_chunk

    def run(self, result=None):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._tmp_dir = tmp_dir

            cfg = {
                "timeout.correlation": 0.2,
                "interval.read.non_event_date": 0.02,
                "interval.read.retry": 0.01
            }
            m = EuXFELInterface(cfg)
            m.add_control_channel(dc.FLOAT32, 'XFEL.A/B/C/D')
            m.add_control_channel(dc.FLOAT32, 'XFEL.A/B/C/E', no_event=True)
            m.add_diagnostic_channel(dc.IMAGE, 'XFEL.H/I/J/K', shape=(3, 4), dtype='uint16')
            self._machine = m

            self._sc = MachineScan(m)

            super().run(result)

    def testSampleDistance(self):
        n = 1
        self._sc.add_param("param1", dist=0.5, lb=10, ub=20)
        self._sc.add_param("param2", dist=0.5, lb=-5, ub=5)
        seq = self._sc._generate_param_sequence(n)
        self.assertEqual(1, len(seq))

        n = 100
        self._sc.add_param("param3", dist=0.3, start=-1., stop=1., num=10)
        seq = self._sc._generate_param_sequence(n)
        self.assertEqual(1000, len(seq))
        for i in range(len(seq) - 1):
            self.assertGreater(abs(seq[i][0] - seq[i+1][0]), 0.5)
            self.assertGreater(abs(seq[i][1] - seq[i+1][1]), 0.5)
            self.assertGreater(abs(seq[i][2] - seq[i+1][2]), 0.3)

        with self.assertRaises(RuntimeError):
            self._sc._params.pop('param3')
            self._sc._param_dists.pop('param3')
            self._sc.add_param("param3", dist=1., start=-1., stop=1., num=10)
            self._sc._generate_param_sequence(n)

    def _prepare_dataset(self):
        m = self._machine
        return {
            "XFEL.A/B/C/D": ddgen.scalar(
                1., m._controls["XFEL.A/B/C/D"].value_schema(), pid=_INITIAL_PID),
            "XFEL.A/B/C/E": ddgen.scalar(
                100., m._controls["XFEL.A/B/C/E"].value_schema(), pid=_INITIAL_PID),
            "XFEL.H/I/J/K": ddgen.image(
                m._diagnostics["XFEL.H/I/J/K"].value_schema(), pid=_INITIAL_PID),
        }

    @patch("liso.experiment.doocs_interface.pydoocs_write")
    @patch("liso.experiment.doocs_interface.pydoocs_read")
    def testRunFolderCreation(self, patched_read, patched_write):
        sc = self._sc
        dataset = self._prepare_dataset()
        patched_read.side_effect = lambda x: _side_effect_read(dataset, x)

        sc.add_param('XFEL.A/B/C/D', lb=-3, ub=3)

        with tempfile.TemporaryDirectory() as tmp_dir:
            sc.scan(2, output_dir=tmp_dir, timeout=0.001)
            sc.scan(2, output_dir=tmp_dir, timeout=0.001)
            path = pathlib.Path(tmp_dir)
            self.assertListEqual([path.joinpath(f'r000{i}') for i in [1, 2]],
                                 sorted((path.iterdir())))

            path.joinpath("r0006").mkdir()
            sc.scan(2, output_dir=tmp_dir, timeout=0.001)
            self.assertListEqual([path.joinpath(f'r000{i}') for i in [1, 2, 6, 7]],
                                 sorted((path.iterdir())))

    @patch("liso.experiment.doocs_interface.pydoocs_write")
    @patch("liso.experiment.doocs_interface.pydoocs_read")
    def testChmod(self, patched_read, patched_write):
        sc = self._sc
        dataset = self._prepare_dataset()
        patched_read.side_effect = lambda x: _side_effect_read(dataset, x)

        sc.add_param('XFEL.A/B/C/D', lb=-3, ub=3)

        with tempfile.TemporaryDirectory() as tmp_dir:
            sc.scan(10, output_dir=tmp_dir, timeout=0.001)
            path = pathlib.Path(tmp_dir).joinpath('r0001')
            for file in path.iterdir():
                self.assertEqual('400', oct(file.stat().st_mode)[-3:])

        with tempfile.TemporaryDirectory() as tmp_dir:
            sc.scan(10, output_dir=tmp_dir, chmod=False, timeout=0.001)
            path = pathlib.Path(tmp_dir).joinpath('r0001')
            for file in path.iterdir():
                self.assertNotEqual('400', oct(file.stat().st_mode)[-3:])

    @patch("liso.experiment.doocs_interface.pydoocs_write")
    @patch("liso.experiment.doocs_interface.pydoocs_read")
    def testScan(self, patched_read, patched_write):
        sc = self._sc
        dataset = self._prepare_dataset()
        patched_read.side_effect = lambda x: _side_effect_read(dataset, x)

        with self.assertRaisesRegex(ValueError, "No scan parameters specified"):
            sc.scan(10)

        with tempfile.TemporaryDirectory() as tmp_dir:
            sc.add_param('XFEL.A/B/C/D', lb=-3, ub=3)
            sc.add_param('XFEL.A/B/C/E', lb=-3, ub=3)

            path = pathlib.Path(tmp_dir)

            n_pulses = 40
            sc.scan(n_pulses, output_dir=tmp_dir, timeout=0.01)

            run = open_run(path.joinpath('r0001'))
            run.info()

            control_data = run.get_controls()

            pids = control_data.index
            self.assertEqual(len(np.unique(pids)), len(pids))
            np.testing.assert_array_equal(
                control_data['XFEL.A/B/C/D'], pids - _INITIAL_PID + 1)
            if len(pids) != 1:
                self.assertEqual(len(np.unique(control_data['XFEL.A/B/C/E'])),
                                 len(control_data['XFEL.A/B/C/E']))

            img_data = run.channel("XFEL.H/I/J/K").numpy()

            self.assertTupleEqual((len(pids), 3, 4), img_data.shape)
            self.assertTrue(np.all(img_data[1] == pids[1] - _INITIAL_PID + 1))
            self.assertTrue(np.all(img_data[-1] == pids[-1] - _INITIAL_PID + 1))

    @patch("liso.experiment.doocs_interface.pydoocs_write")
    @patch("liso.experiment.doocs_interface.pydoocs_read")
    def testLogInitialParameters(self, patched_read, patched_write):
        sc = self._sc
        dataset = self._prepare_dataset()

        with tempfile.TemporaryDirectory() as tmp_dir:
            sc.add_param('XFEL.A/B/C/D', lb=-3, ub=3)
            sc.add_param('XFEL.A/B/C/E', lb=-3, ub=3)

            patched_read.side_effect = lambda x: _side_effect_read(dataset, x)
            sc.scan(10, output_dir=tmp_dir, timeout=0.01)

            def _side_effect_raise(x):
                raise DoocsException
            patched_read.side_effect = _side_effect_raise
            with self.assertRaisesRegex(RuntimeError,
                                        "Failed to read all the initial values"):
                sc.scan(10, output_dir=tmp_dir, timeout=0.01)
