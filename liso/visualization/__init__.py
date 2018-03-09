import matplotlib
matplotlib.use('Qt5Agg')

from .phasespace_plot import AstraPhaseSpacePlot
from .phasespace_plot import ImpacttPhaseSpacePlot
from .line_plot import AstraLinePlot
from .line_plot import ImpacttLinePlot
from .phasespace_gui import phasespace_gui

__all__ = [
    'AstraPhaseSpacePlot',
    'ImpacttPhaseSpacePlot',
    'AstraLinePlot',
    'ImpacttLinePlot',
    'phasespace_gui'
]