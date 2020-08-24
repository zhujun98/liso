"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import os


def check_data_file(filepath):
    """Check the status of a data file."""
    if not os.path.isfile(filepath):
        raise RuntimeError(filepath + " does not exist!")
    if not os.path.getsize(filepath):
        raise RuntimeError(filepath + " is empty!")
