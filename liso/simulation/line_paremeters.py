#!/usr/bin/python

"""
Hold three classes:

- BeamEvolution:
    Store and update the beam evolution and its statistics

- Stats
    Store and update the statistics of a single variable

Important note:
_______________
The calculated Twiss parameters are based on the canonical coordinates
of the beam. Therefore, when the beam has a large energy spread or a
big divergence angle, the result will be quite different from the
true Twiss parameters. However, one can set TR_EmitS = .T in ASTRA to
get the very-close Twiss parameters even in extreme conditions.


Author: Jun Zhu

"""
from .stats import Stats
from ..backend import config

V_LIGHT = config['vLight']
M_E = config['me']
Q_E = config['qe']
INF = config['INF']

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
    def __init__(self, rootname, zlim=None, opt=False):
        """Initialize BeamStats object

        :param root_name: string
            The root name of the output files. For Impact-T files,
            root_name will be set to 'fort' if not given.
        :param zlim: scalar/tuple
            If None, passed as (-INF, INF)
            if scalar, being passed as (left, INF)
            if tuple, the first two elements being passed as (left, right)
        """
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
        """Print output"""
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
