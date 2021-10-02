# pylint: disable=attribute-defined-outside-init
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from liso import EuXFELInterface, MachineScan, open_run
from liso import doocs_channels as dc
from liso.exceptions import LisoRuntimeError
from liso.experiment.doocs_interface import DoocsException
from liso.io import ExpWriter
from liso.logging import logger
from ...experiment.tests import DoocsDataGenerator as ddgen

_INITIAL_PID = 1000


def _side_effect_read(dataset, address):
    data = deepcopy(dataset[address])
    if 'write' not in address and data['macropulse'] >= _INITIAL_PID:
        dataset[address]['data'] += 1
        dataset[address]['macropulse'] += 1
    return data


class TestMachineScan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._orig_image_chunk = ExpWriter._IMAGE_CHUNK
        ExpWriter._IMAGE_CHUNK = (3, 2)

    @classmethod
    def tearDownClass(cls):
        ExpWriter._IMAGE_CHUNK = cls._orig_image_chunk

    def setUp(self):
        logger.setLevel('ERROR')

    def run(self, result=None):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._tmp_dir = tmp_dir

            cfg = {
                "timeout.correlation": 0.1,
                "interval.read.retry": 0.005
            }
            m = EuXFELInterface(cfg)
            m._validation_prob = 1.
            m.add_control_channel('XFEL.A/B/C/D', dc.FLOAT32,
                                  write_address='XFEL.A/B/C/D.write')
            m.add_control_channel('XFEL.A/B/C/E', dc.INT64, non_event=True)
            m.add_diagnostic_channel('XFEL.H/I/J/K', dc.ARRAY, shape=(3, 4), dtype='uint16')
            m.add_diagnostic_channel('XFEL.H/I/J/L', dc.ARRAY, shape=(100,), dtype='float64')
            self._machine = m

            self._sc = MachineScan(m, read_delay=0.)

            super().run(result)

    def _prepare_dataset(self):
        m = self._machine
        return {
            "XFEL.A/B/C/D": ddgen.scalar(
                1., m._channels["XFEL.A/B/C/D"].value_schema(), pid=_INITIAL_PID),
            "XFEL.A/B/C/D.write": ddgen.scalar(
                1., m._channels["XFEL.A/B/C/D"].value_schema(), pid=_INITIAL_PID),
            "XFEL.A/B/C/E": ddgen.scalar(
                100, m._channels["XFEL.A/B/C/E"].value_schema(), pid=_INITIAL_PID),
            "XFEL.H/I/J/K": ddgen.array(
                m._channels["XFEL.H/I/J/K"].value_schema(), pid=_INITIAL_PID),
            "XFEL.H/I/J/L": ddgen.array(
                m._channels["XFEL.H/I/J/L"].value_schema(), pid=_INITIAL_PID),
        }

    def testInitialization(self):
        m = EuXFELInterface()
        with self.assertRaisesRegex(ValueError, "not a valid scan policy"):
            MachineScan(m, policy='A')

        m = self._machine
        assert m.control_channels == {'XFEL.A/B/C/D', 'XFEL.A/B/C/E'}
        assert m.diagnostic_channels == {'XFEL.H/I/J/K', 'XFEL.H/I/J/L'}
        assert m._event == {'XFEL.A/B/C/D', 'XFEL.H/I/J/K', 'XFEL.H/I/J/L'}
        assert m._non_event == {'XFEL.A/B/C/E'}

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

        with self.assertRaises(ValueError):
            self._sc._params.pop('param3')
            self._sc._param_dists.pop('param3')
            self._sc.add_param("param3", dist=1., start=-1., stop=1., num=10)
            self._sc._generate_param_sequence(n)

    @patch("liso.experiment.doocs_interface.pydoocs_write")
    @patch("liso.experiment.doocs_interface.pydoocs_read")
    def testRunFolderCreation(self, patched_read, _):
        sc = self._sc
        dataset = self._prepare_dataset()
        patched_read.side_effect = lambda x: _side_effect_read(dataset, x)

        sc.add_param('XFEL.A/B/C/D', lb=-3, ub=3)

        with tempfile.TemporaryDirectory() as tmp_dir:
            sc.scan(2, output_dir=tmp_dir)
            sc.scan(2, output_dir=tmp_dir)
            path = Path(tmp_dir)
            self.assertListEqual([path.joinpath(f'r000{i}') for i in [1, 2]],
                                 sorted((path.iterdir())))

            path.joinpath("r0006").mkdir()
            sc.scan(2, output_dir=tmp_dir)
            self.assertListEqual([path.joinpath(f'r000{i}') for i in [1, 2, 6, 7]],
                                 sorted((path.iterdir())))

    @patch("liso.experiment.doocs_interface.pydoocs_write")
    @patch("liso.experiment.doocs_interface.pydoocs_read")
    def testChmod(self, patched_read, _):
        sc = self._sc
        dataset = self._prepare_dataset()
        patched_read.side_effect = lambda x: _side_effect_read(dataset, x)

        sc.add_param('XFEL.A/B/C/D', lb=-3, ub=3)

        with tempfile.TemporaryDirectory() as tmp_dir:
            sc.scan(5, output_dir=tmp_dir)
            path = Path(tmp_dir).joinpath('r0001')
            for file in path.iterdir():
                self.assertEqual('400', oct(file.stat().st_mode)[-3:])

        with tempfile.TemporaryDirectory() as tmp_dir:
            sc.scan(5, output_dir=tmp_dir, chmod=False)
            path = Path(tmp_dir).joinpath('r0001')
            for file in path.iterdir():
                self.assertNotEqual('400', oct(file.stat().st_mode)[-3:])

    @patch("liso.experiment.doocs_interface.pydoocs_write")
    @patch("liso.experiment.doocs_interface.pydoocs_read")
    def testSafeScan(self, mocked_read, mocked_write):
        logger.setLevel("INFO")
        sc = self._sc
        sc.add_param('XFEL.A/B/C/D', lb=-3, ub=3)

        with tempfile.TemporaryDirectory() as tmp_dir:
            def _side_effect_raise(_):
                raise DoocsException
            mocked_read.side_effect = _side_effect_raise
            with self.assertRaisesRegex(LisoRuntimeError, "XFEL.A/B/C/D"):
                sc.scan(1, output_dir=tmp_dir)

        dataset = self._prepare_dataset()
        mocked_read.side_effect = lambda x: _side_effect_read(dataset, x)
        with self.assertLogs(level="INFO") as cm:
            with tempfile.TemporaryDirectory() as tmp_dir:
                sc.scan(1, output_dir=tmp_dir)
        assert mocked_write.call_count == 2
        assert mocked_write.call_args_list[1][0] == ('XFEL.A/B/C/D.write', 1.0)
        assert "Initial machine setup: " in cm.output[-4]
        assert "Machine setup restored: " in cm.output[-2]

    @patch("liso.experiment.doocs_interface.pydoocs_write")
    @patch("liso.experiment.doocs_interface.pydoocs_read")
    def testScan(self, patched_read, _):
        sc = self._sc
        dataset = self._prepare_dataset()
        patched_read.side_effect = lambda x: _side_effect_read(dataset, x)

        with self.assertRaisesRegex(ValueError, "No scan parameters specified"):
            sc.scan(1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            sc.add_param('XFEL.A/B/C/D', lb=-3, ub=3)
            sc.add_param('XFEL.A/B/C/E', lb=-3, ub=3)

            n_pulses = 20
            sc.scan(n_pulses, output_dir=tmp_dir)
            run = open_run(tmp_dir.joinpath('r0001'))
            run.info()

            pids = run.pulse_ids
            assert pids[0] == 1000
            assert len(np.unique(pids)) == len(pids)

            control_data = run.get_controls()
            np.testing.assert_array_equal(
                control_data['XFEL.A/B/C/D'], pids - _INITIAL_PID + 1)

            img_data = run.channel("XFEL.H/I/J/K").numpy()

            assert (len(pids), 3, 4) == img_data.shape
            self.assertTrue(np.all(img_data[0] == pids[0] - _INITIAL_PID + 1))
            self.assertTrue(np.all(img_data[-1] == pids[-1] - _INITIAL_PID + 1))

            array_data = run.channel("XFEL.H/I/J/L").numpy()
            assert (len(pids), 100) == array_data.shape

    @patch("liso.experiment.doocs_interface.pydoocs_write")
    @patch("liso.experiment.doocs_interface.pydoocs_read")
    def testValidationDuringScan(self, mocked_read, _):
        sc = self._sc
        dataset = self._prepare_dataset()
        mocked_read.side_effect = lambda x: _side_effect_read(dataset, x)

        with tempfile.TemporaryDirectory() as tmp_dir:
            sc.add_param('XFEL.A/B/C/D', lb=-3, ub=3)
            sc.add_param('XFEL.A/B/C/E', lb=-3, ub=3)

            dataset["XFEL.A/B/C/E"]['data'] = 1.0
            with self.assertRaisesRegex(TypeError, "Validation error"):
                sc.scan(1, output_dir=tmp_dir)

    @patch("liso.experiment.doocs_interface.pydoocs_write")
    @patch("liso.experiment.doocs_interface.pydoocs_read")
    def testScanWithMoreThanOneRead1(self, patched_read, _):
        sc = self._sc
        sc._n_reads = 4
        dataset = self._prepare_dataset()
        patched_read.side_effect = lambda x: _side_effect_read(dataset, x)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            sc.add_param('XFEL.A/B/C/D', lb=-3, ub=3)
            sc.add_param('XFEL.A/B/C/E', lb=-3, ub=3)

            n_pulses = 20
            sc.scan(n_pulses, output_dir=tmp_dir)
            run = open_run(tmp_dir.joinpath('r0001'))
            run.info()

            pids = run.pulse_ids
            self.assertListEqual(list(pids[:4]), [1000, 1001, 1002, 1003])
            assert len(np.unique(pids)) == len(pids) == 80

    @patch("liso.experiment.doocs_interface.pydoocs_write")
    @patch("liso.experiment.doocs_interface.pydoocs_read")
    def testScanWithMoreThanOneReadFail(self, patched_read, _):
        logger.setLevel('WARNING')

        sc = self._sc
        sc._n_reads = 4
        dataset = self._prepare_dataset()

        def _side_effect_read_capped(dataset, address):
            data = deepcopy(dataset[address])
            if _INITIAL_PID + 2 > data['macropulse'] >= _INITIAL_PID:
                dataset[address]['macropulse'] += 1
            return data

        patched_read.side_effect = lambda x: _side_effect_read_capped(dataset, x)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            sc.add_param('XFEL.A/B/C/D', lb=-3, ub=3)
            sc.add_param('XFEL.A/B/C/E', lb=-3, ub=3)

            n_pulses = 20
            with self.assertLogs(level="WARNING") as cm:
                sc.scan(n_pulses, output_dir=tmp_dir)
            assert "Failed to readout 4 data" in cm.output[0]

            with self.assertRaisesRegex(Exception, "No HDF5 files"):
                open_run(tmp_dir.joinpath('r0001'))
