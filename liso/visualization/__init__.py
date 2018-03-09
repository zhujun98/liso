import matplotlib
matplotlib.use('Qt5Agg')

from .phasespace_plot import AstraPhaseSpacePlot
from .phasespace_plot import ImpacttPhaseSpacePlot
from .line_plot import AstraLinePlot
from .line_plot import ImpacttLinePlot
from .main_gui import main_gui


__all__ = [
    'AstraPhaseSpacePlot',
    'ImpacttPhaseSpacePlot',
    'AstraLinePlot',
    'ImpacttLinePlot',
    'main_gui'
]