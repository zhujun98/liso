from unittest.mock import patch
from argparse import Namespace
import tempfile

import pytest

from liso.experiment.doocs_app import monitor


@patch("time.sleep", side_effect=KeyboardInterrupt)
@patch("liso.experiment.doocs_interface.DoocsInterface.monitor")
def test_monitor(mocked_monitor, _):
    with patch("argparse.ArgumentParser.parse_args",
               return_value=Namespace(channels=None, file=None, correlate=False)):
        with pytest.raises(ValueError, match="No DOOCS channel"):
            monitor()

    with patch("argparse.ArgumentParser.parse_args",
               return_value=Namespace(channels="A/B/C/D", file=None, correlate=False)):
        monitor()
        mocked_monitor.assert_called_once_with(correlate=False, app=True)
        mocked_monitor.reset_mock()

    with patch("argparse.ArgumentParser.parse_args",
               return_value=Namespace(channels="A/B/C/D", file=None, correlate=True)):
        monitor()
        mocked_monitor.assert_called_once_with(correlate=True, app=True)
        mocked_monitor.reset_mock()

    ch_gt = ["A/B/C/D", "A/B/C/E", "A/B/C/F", "A/B/C/G"]
    # pylint: disable=line-too-long
    with patch("liso.experiment.doocs_interface.DoocsInterface.add_diagnostic_channel") as mocked_add:
        with tempfile.NamedTemporaryFile("w+") as f:
            f.writelines([v + "\n" for v in ch_gt[1:]])
            f.seek(0)
            with patch("argparse.ArgumentParser.parse_args",
                       return_value=Namespace(channels="A/B/C/D", file=f.name, correlate=False)):
                monitor()
                for i, ch in enumerate(ch_gt):
                    assert (ch,) == mocked_add.call_args_list[i][0]
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
                    assert (ch,) == mocked_add.call_args_list[i][0]
                for i, ch in enumerate(ch_gt):
                    if i < 2:
                        assert {} == mocked_add.call_args_list[i][1]
                    else:
                        assert {'non_event': True} == mocked_add.call_args_list[i][1]
                mocked_add.reset_mock()
