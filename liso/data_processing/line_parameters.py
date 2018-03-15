#!/usr/bin/python
"""
Author: Jun Zhu

Important note:
_______________
The calculated Twiss parameters are based on the canonical coordinates
of the beam. Therefore, when the beam has a large energy spread or a
big divergence angle, the result will be quite different from the
true Twiss parameters. However, one can set TR_EmitS = .T in ASTRA to
get the very-close Twiss parameters even in extreme conditions.
"""
import numpy as np

from ..config import Config

V_LIGHT = Config.vLight
M_E = Config.me
Q_E = Config.qe
INF = Config.INF

CONST_E = M_E*V_LIGHT**2/Q_E


class LineParameters(object):
    """Store the beam evolution and its statistics

    Attributes
    ----------
    z: Stats object
        Longitudinal position (m).
    gamma: Stats object
        Lorentz factor.
    SdE: Stats object
        RMS energy spread (eV).
    Sx/Sy/Sz: Stats objects
        RMS bunch sizes (m).
    betax/betay: Stats objects
        Beta functions (m).
    alphax/alphay: Stats objects
        Alpha functions.
    emitx/emity: Stats objects
        Normalized canonical emittance (m.rad).
    emitx_tr/emity_tr: Stats objects
        Normalized trace-space emittance (m.rad).
    """
    def __init__(self):
        """Initialization."""
        self.z = Stats()
        self.gamma = Stats()
        self.SdE = Stats()
        self.Sx = Stats()
        self.Sy = Stats()
        self.Sz = Stats()
        self.betax = Stats()
        self.betay = Stats()
        self.alphax = Stats()
        self.alphay = Stats()
        self.emitx = Stats()
        self.emity = Stats()
        self.emitx_tr = Stats()
        self.emity_tr = Stats()

    def __str__(self):
        text = '- z (m)\n'
        text += str(self.__getattribute__('z'))
        text += '- gamma\n'
        text += str(self.__getattribute__('gamma'))
        text += '- SdE (eV)\n'
        text += str(self.__getattribute__('SdE'))
        text += '- Sx (m)\n'
        text += str(self.__getattribute__('Sx'))
        text += '- Sy (m)\n'
        text += str(self.__getattribute__('Sy'))
        text += '- Sz (m)\n'
        text += str(self.__getattribute__('Sz'))
        text += '- betax (m)\n'
        text += str(self.__getattribute__('betax'))
        text += '- betay (m)\n'
        text += str(self.__getattribute__('betay'))
        text += '- alphax (m)\n'
        text += str(self.__getattribute__('alphax'))
        text += '- alphay (m)\n'
        text += str(self.__getattribute__('alphay'))
        text += '- emitx (m.rad)\n'
        text += str(self.__getattribute__('emitx'))
        text += '- emity (m.rad)\n'
        text += str(self.__getattribute__('emity'))
        text += '- emitx_tr (m.rad)\n'
        text += str(self.__getattribute__('emitx_tr'))
        text += '- emity_tr (m.rad)\n'
        text += str(self.__getattribute__('emity_tr'))

        return text


class Stats(object):
    """Store the statistic values of an array-like object.

    Attributes
    ----------
    start: float
        First value.
    end: float
        Last value.
    max: float
        Maximum value.
    min: float
        Minimum value.
    ave: float
        Average value.
    std: float
        Standard deviation.
    """
    def __init__(self):
        """Initialization."""
        self.start = None
        self.end = None
        self.max = None
        self.min = None
        self.ave = None
        self.std = None

    def __str__(self):
        text = "{:12}    {:12}    {:12}    {:12}    {:12}    {:12}\n".\
            format('start', 'end', 'minimum', 'maximum', 'average', 'std')

        text += "{:12.4e}    {:12.4e}    {:12.4e}    {:12.4e}    {:12.4e}    {:12.4e}\n\n".\
            format(self.start, self.end, self.min, self.max, self.ave, self.std)

        return text

    def update(self, data):
        """Update attributes

        :param data: array-like
            Input data.
        """
        data = np.asarray(data)
        if data.ndim > 1:
            raise ValueError("One-dimensional array is foreseen!")

        self.start = data[0]
        self.end = data[-1]
        self.max = data.max()
        self.min = data.min()
        self.ave = data.mean()
        self.std = data.std(ddof=0)
