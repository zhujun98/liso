__all__ = []

try:
    import matplotlib
    try:
        matplotlib.use('Qt5Agg')
    except ImportError:
        print("Failed to use 'Qt5Agg'. Use the default setting.")

    from .phasespace_plot import PhaseSpacePlot
    from .line_plot import LinePlot
    __all__ += ['PhaseSpacePlot', 'LinePlot']

except ImportError:
    print("Please install matplotlib to use PhaseSpacePlot and LinePlot!")

try:
    from .main_gui import main_gui as gui
    __all__ = ['gui']
except ImportError:
    print("Please install PyQt5 and pyqtgraph to use GUI!")

