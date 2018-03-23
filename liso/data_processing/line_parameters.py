#!/usr/bin/python
"""
Author: Jun Zhu
"""


class LineParameters(object):
    """Store a certain statistic of beam parameters."""
    def __init__(self):
        """Initialization."""
        self.z = 0.0  # longitudinal position (m)
        self.gamma = 0.0  # Lorentz factor
        self.SdE = 0.0  # rms energy spread (eV)
        self.Sx = 0.0  # rms bunch sizes (m)
        self.Sy = 0.0  # rms bunch sizes (m)
        self.Sz = 0.0  # rms bunch sizes (m)
        self.betax = 0.0  # beta functions (m)
        self.betay = 0.0  # beta functions (m)
        self.alphax = 0.0  # alpha functions
        self.alphay = 0.0  # alpha functions
        self.emitx = 0.0  # normalized canonical emittance (m.rad)
        self.emity = 0.0  # normalized canonical emittance (m.rad)
        self.emitx_tr = 0.0  # normalized trace-space emittance (m.rad)
        self.emity_tr = 0.0  # normalized trace-space emittance (m.rad)

    def __str__(self):
        text = '\n'
        text += 'z (m): {:11.4e}\n'.format(self.z)
        text += 'gamma: {:11.4e}\n'.format(self.gamma)
        text += 'SdE (eV): {:11.4e}\n'.format(self.SdE)
        text += 'Sx (m): {:11.4e}\n'.format(self.Sx)
        text += 'Sy (m): {:11.4e}\n'.format(self.Sy)
        text += 'Sz (m): {:11.4e}\n'.format(self.Sz)
        text += 'betax (m): {:11.4e}\n'.format(self.betax)
        text += 'betay (m): {:11.4e}\n'.format(self.betay)
        text += 'alphax (m): {:11.4e}\n'.format(self.alphax)
        text += 'alphay (m): {:11.4e}\n'.format(self.alphay)
        text += 'emitx (m): {:11.4e}\n'.format(self.emitx)
        text += 'emity (m): {:11.4e}\n'.format(self.emity)
        text += 'emitx_tr (m): {:11.4e}\n'.format(self.emitx_tr)
        text += 'emity_tr (m): {:11.4e}\n'.format(self.emity_tr)

        return text
