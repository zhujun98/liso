"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
from collections.abc import Mapping


class OutputData(Mapping):
    def __init__(self, inputs, phasespaces):
        """Initialization.

        :param dict inputs: input parameters.
        :param dict phasespaces: output phasespaces.
        """
        super().__init__()

        metadata = {
            'input': set(),
            'phasespace': set(),
        }
        for item in inputs:
            metadata['input'].add(item)
        for ps in phasespaces:
            metadata['phasespace'].add(ps)

        self._data = {
            "metadata": metadata,
            "input": inputs,
            "phasespace": phasespaces,
        }

    def __getitem__(self, item):
        """Override."""
        return self._data[item]

    def __iter__(self):
        """Override."""
        return self._data.__iter__()

    def __len__(self):
        """Override."""
        return self._data.__len__()
