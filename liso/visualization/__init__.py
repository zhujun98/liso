from ..logging import create_logger


logger = create_logger(__name__)

__all__ = []

try:
    import matplotlib
    try:
        matplotlib.use('Qt5Agg')
    except ImportError:
        logger.warning("Failed to use 'Qt5Agg'. Use the default setting.")

    from .phasespace_plot import PhaseSpacePlot
    from .line_plot import LinePlot
    __all__ += ['PhaseSpacePlot', 'LinePlot']

except ImportError:
    logger.warning(
        "Please install matplotlib to use PhaseSpacePlot and LinePlot!")

try:
    from .main_gui import main_gui as gui
    __all__ = ['gui']
except ImportError:
    logger.warning("Please install PyQt5 and pyqtgraph to use GUI!")
