# pylint: disable=attribute-defined-outside-init
import asyncio
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from liso import EuXFELInterface, MachineScan, open_run
from liso import doocs_channels as dc
from liso.experiment.doocs_interface import DoocsException
from liso.io import ExpWriter
from liso.logging import logger
from ...experiment.tests import DoocsDataGenerator as ddgen

logger.setLevel('ERROR')

_INITIAL_PID = 1000


def _side_effect_read(dataset, address):
    data = deepcopy(dataset[address])
    if data['macropulse'] >= _INITIAL_PID:
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
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

    def tearDown(self):
        asyncio.set_event_loop(None)
        self._loop.close()

    def run(self, result=None):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._tmp_dir = tmp_dir

            cfg = {
                "timeout.correlation": 0.02,
                "interval.read.retry": 0.002
            }
            m = EuXFELInterface(cfg)
            m.add_control_channel('XFEL.A/B/C/D', dc.FLOAT32)
            m.add_control_channel('XFEL.A/B/C/E', dc.FLOAT32, non_event=True)
            m.add_diagnostic_channel('XFEL.H/I/J/K', dc.IMAGE, shape=(3, 4), dtype='uint16')
            self._machine = m

            self._sc = MachineScan(m, read_delay=0.)

            super().run(result)

    def _prepare_dataset(self):
        m = self._machine
        return {
            "XFEL.A/B/C/D": ddgen.scalar(
                1., m._channels["XFEL.A/B/C/D"].value_schema(), pid=_INITIAL_PID),
            "XFEL.A/B/C/E": ddgen.scalar(
                100., m._channels["XFEL.A/B/C/E"].value_schema(), pid=_INITIAL_PID),
            "XFEL.H/I/J/K": ddgen.image(
                m._channels["XFEL.H/I/J/K"].value_schema(), pid=_INITIAL_PID),
        }

    def testInitialization(self):
        m = EuXFELInterface()
        with self.assertRaisesRegex(ValueError, "not a valid scan policy"):
            MachineScan(m, policy='A')

        m = self._machine
        assert m.control_channels == {'XFEL.A/B/C/D', 'XFEL.A/B/C/E'}
        assert m.diagnostic_channels == {'XFEL.H/I/J/K'}
        assert m._event == {'XFEL.A/B/C/D', 'XFEL.H/I/J/K'}
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
    def testLogInitialParameters(self, mocked_read, _):
        sc = self._sc
        sc.add_param('XFEL.A/B/C/D', lb=-3, ub=3)

        def _side_effect_raise(_):
            raise DoocsException
        mocked_read.side_effect = _side_effect_raise
        with self.assertRaisesRegex(RuntimeError,
                                    "Failed to read all the initial values"):
            sc.scan(1)

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
            self.assertListEqual(list(pids[:4]), [1002, 1004, 1006, 1008])
            assert len(np.unique(pids)) == len(pids)

            control_data = run.get_controls()
            np.testing.assert_array_equal(
                control_data['XFEL.A/B/C/D'], pids - _INITIAL_PID + 1)

            img_data = run.channel("XFEL.H/I/J/K").numpy()

            assert (len(pids), 3, 4) == img_data.shape
            self.assertTrue(np.all(img_data[0] == pids[0] - _INITIAL_PID + 1))
            self.assertTrue(np.all(img_data[-1] == pids[-1] - _INITIAL_PID + 1))

    @patch("liso.experiment.doocs_interface.pydoocs_write")
    @patch("liso.experiment.doocs_interface.pydoocs_read")
    def testScanWithMoreThanOneRead(self, patched_read, _):
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
            self.assertListEqual(
                list(pids[:8]), [1002, 1003, 1004, 1005, 1007, 1008, 1009, 1010])
            assert len(np.unique(pids)) == len(pids)
