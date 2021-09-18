import sys
from unittest.mock import patch
from argparse import Namespace
import tempfile

import pytest

from liso.experiment.doocs_app import monitor
from liso.experiment.doocs_channels import AnyDoocsChannel


@patch("time.sleep", side_effect=KeyboardInterrupt)
@patch("liso.experiment.doocs_interface.DoocsInterface.read",
       return_value=(None, dict(), dict()))
def test_monitor(patched_read, patched_sleep):
    with patch("argparse.ArgumentParser.parse_args",
               return_value=Namespace(channels=None, file=None, correlate=False)):
        with pytest.raises(ValueError, match="No DOOCS channel"):
            monitor()

    with patch("argparse.ArgumentParser.parse_args",
               return_value=Namespace(channels="A/B/C/D", file=None, correlate=False)):
        monitor()
        patched_read.assert_called_once()
        assert patched_read.call_args_list[0][1] == {'correlate': False}
        patched_read.reset_mock()

    ch_gt = ["A/B/C/D", "A/B/C/E", "A/B/C/F", "A/B/C/G"]
    with patch("liso.experiment.doocs_interface.DoocsInterface.add_diagnostic_channel") as mocked_add:
        with tempfile.NamedTemporaryFile("w+") as f:
            f.writelines([v + "\n" for v in ch_gt[1:]])
            f.seek(0)
            with patch("argparse.ArgumentParser.parse_args",
                       return_value=Namespace(channels="A/B/C/D", file=f.name, correlate=False)):
                monitor()
                for i, ch in enumerate(ch_gt):
                    assert (AnyDoocsChannel, ch) == mocked_add.call_args_list[i][0]
                mocked_add.reset_mock()

        with tempfile.NamedTemporaryFile("w+") as f:
            f.write(ch_gt[1] + " \n")
            f.write(" ---xxxxx\n")
            f.writelines([v + "\n" for v in ch_gt[2:]])
            f.seek(0)
            with patch("argparse.ArgumentParser.parse_args",
                       return_value=Namespace(channels="A/B/C/D", file=f.name, correlate=False)):
                monitor()
                for i, ch in enumerate(ch_gt):
                    assert (AnyDoocsChannel, ch) == mocked_add.call_args_list[i][0]
                for i, ch in enumerate(ch_gt):
                    if i < 2:
                        assert {} == mocked_add.call_args_list[i][1]
                    else:
                        assert {'non_event': True} == mocked_add.call_args_list[i][1]
                mocked_add.reset_mock()
