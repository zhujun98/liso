"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
from ..exceptions import *


class BeamParameters(object):
    """Beam parameters for a given distribution.

    This class serves as a structure only used to store data.
    """
    def __init__(self):
        """Initialization."""
        self._n = 0
        self._q = 0.0
        self._charge = 0.0

        self.p = 0.0
        self.gamma = 0.0

        self.Sx = 0.0
        self.Sy = 0.0
        self.St = 0.0
        self.Sz = 0.0
        self.Sdelta = 0.0
        self.Sdelta_un = 0.0
        self.chirp = 0.0
        self.I_peak = 0.0

        self.emitx = 0.0
        self.emity = 0.0
        self.emitx_tr = 0.0
        self.emity_tr = 0.0

        self.betax = 0.0
        self.betay = 0.0
        self.alphax = 0.0
        self.alphay = 0.0

        self.emitx_slice = 0.0
        self.emity_slice = 0.0
        self.Sdelta_slice = 0.0
        self.dt_slice = 0.0

        self.Cx = 0.0
        self.Cy = 0.0
        self.Cxp = 0.0
        self.Cyp = 0.0
        self.Ct = 0.0

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        if not isinstance(value, int):
            raise TypeError("The number of particles must be an integer!")
        self._n = value
        self._charge = self._n * self._q

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, value):
        if self.n == 0 and value != 0.0:
            raise BeamParametersInconsistentError(
                "Zero particle with non-zero charge!")
        self._q = value
        self._charge = self.n * self._q

    @property
    def charge(self):
        return self._charge

    @charge.setter
    def charge(self, value):
        if self.n == 0:
            if value != 0.0:
                raise BeamParametersInconsistentError(
                    "Zero particle with non-zero charge!")
            else:
                self._charge = 0.0
                self._q = 0.0
                return

        self._charge = value
        self._q = value / self.n

    def __str__(self):
        text = "{:16}    {:16}    {:16}    {:16}\n". \
            format('n', 'charge (C)', 'p', 'I_peak (A)')
        text += "{:16.4e}    {:16.4e}    {:16.4e}    {:16.4e}\n\n". \
            format(self.n, self.q, self.p, self.I_peak)
        text += "{:16}    {:16}    {:16}    {:16}\n". \
            format('emitx (m)', 'emity (m)', 'Sx (m)', 'Sy (m)')
        text += "{:16.4e}    {:16.4e}    {:16.4e}    {:16.4e}\n\n". \
            format(self.emitx, self.emity, self.Sx, self.Sy)
        text += "{:16}    {:16}    {:16}    {:16}\n". \
            format('betax (m)', 'betay (m)', 'alphax', 'alphay')
        text += "{:16.4e}    {:16.4e}    {:16.4e}    {:16.4e}\n\n". \
            format(self.betax, self.betay, self.alphax, self.alphay)
        text += "{:16}    {:16}    {:16}    {:16}\n". \
            format('St (s)', 'Sdelta', 'chirp (1/m)', 'Ct (s)')
        text += "{:16.4e}    {:16.4e}    {:16.4e}    {:16.4e}\n\n". \
            format(self.St, self.Sdelta, self.chirp, self.Ct)
        text += "{:16}    {:16}    {:16}    {:16}\n". \
            format('emitx_slice (m)', 'emity_slice (m)', 'Sdelta_slice', 'dt_slice (s)')
        text += "{:16.4e}    {:16.4e}    {:16.4e}    {:16.4e}\n\n". \
            format(self.emitx_slice, self.emity_slice, self.Sdelta_slice, self.dt_slice)
        text += "{:16}    {:16}    {:16}    {:16}\n". \
            format('Cx (m)', 'Cy (m)', 'Cxp (rad)', 'Cyp (rad)')
        text += "{:16.4e}    {:16.4e}    {:16.4e}    {:16.4e}\n\n". \
            format(self.Cx, self.Cy, self.Cxp, self.Cyp)
        text += "{:16}    {:16}    {:16}\n". \
            format('emitx_tr (m)', 'emity_tr (m)', 'Sdelta_un')
        text += "{:16.4e}    {:16.4e}    {:16.4e}\n". \
            format(self.emitx_tr, self.emity_tr, self.Sdelta_un)

        return text
