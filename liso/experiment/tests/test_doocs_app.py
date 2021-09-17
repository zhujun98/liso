import sys
from unittest.mock import patch
from argparse import Namespace

import pytest

from liso.experiment.doocs_app import monitor


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
        assert patched_read.call_args_list[0][1] == {'correlate': False, 'validate': False}
