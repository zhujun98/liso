"""
Classes for plotting phase-spaces from different codes.

Author: Jun Zhu

"""
import os
from abc import abstractmethod

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from ..data_processing import parse_astra_phasespace
from ..data_processing import parse_impactt_phasespace
from ..data_processing import parse_impactz_phasespace
from ..data_processing import parse_genesis_phasespace
from ..simulation.beam_parameters import BeamParameters
from .vis_utils import get_default_unit
from .vis_utils import get_unit_scale
from .vis_utils import get_label
from .vis_utils import sample_data


LABEL_FONT_SIZE = 26
TITLE_FONT_SIZE = 18
TICK_FONT_SIZE = 20
LEGEND_FONT_SIZE = 18
LABEL_PAD = 8
TICK_PAD = 8
MAX_LOCATOR = 7
AX_MARGIN = 0.05


class PhaseSpacePlot(object):
    """Plot the beam phase-space."""
    def __init__(self, pfile, **kwargs):
        """Initialization

        :param pfile: string
            Path name of the particle file.
        """
        self.pfile = pfile
        self.charge = None
        self.data = self._load_data()
        self.params = BeamParameters()
        self._options = ['x', 'y', 'z', 'xp', 'yp', 't', 'p']

    @abstractmethod
    def _load_data(self):
        raise NotImplemented

    def cloud(self, var_x, var_y, **kwargs):
        self._plot(var_x, var_y, **kwargs)

    def save_cloud(self, var_x, var_y, **kwargs):
        self._plot(var_x, var_y, save_image=True,  **kwargs)

    def scatter(self, var_x, var_y, **kwargs):
        self._plot(var_x, var_y, cloud_plot=False, **kwargs)

    def save_scatter(self, var_x, var_y, **kwargs):
        self._plot(var_x, var_y, cloud_plot=False, save_image=True, **kwargs)

    def _plot(self, var_x, var_y, *,
              ax=None,
              x_unit=None,
              y_unit=None,
              xlim=None,
              ylim=None,
              cloud_plot=True,
              ms=2,
              mc='dodgerblue',
              alpha=1.0,
              bins_2d=500,
              sigma_2d=5,
              sample=20000,
              save_image=False,
              filename='',
              dpi=300,
              show_parameters=True):
        """Show a phase-space on screen or in a file.

        :param var_x: string
            Variable for x-axis.
        :param var_y: string
            Variable for y-axis.
        :param x_unit: string
            Unit for x axis.
        :param y_unit: string
            Unit for y axis.
        :param xlim: tuple, (x_min, x_max)
            Range of the x axis.
        :param ylim: tuple, (y_min, y_max)
            Range of the y axis.
        :param cloud_plot: Boolean
            True for colorful density plot.
        :param ms: int
            Marker size for scatter plots..
        :param mc: string
            Color of markers for non-density plot.
        :param alpha: float, [0, 1]
            Alpha value (transparency). Default = 1.0.
        :param bins_2d: int or [int, int]
            No. of bins used in numpy.histogram2d.
        :param sigma_2d: int/float
            Standard deviation of Gaussian kernel of the Gaussian filter.
        :param sample: non-negative int/float
            If sample < 1.0, sample by fraction else sample by count
            (round to integer).
        :param save_image: bool
            Save image to file. Default = False.
        :param filename: string
            Filename of the output file. Default = ''.
            The file will be saved in the same directory as the particle file.
        :param dpi: int
            DPI of the plot. Default = 300.
        :param show_parameters: bool
            Show beam parameters in the title. Default = True.
        """
        var_x = var_x.lower()
        var_y = var_y.lower()
        if var_x not in self._options or var_y not in self._options:
            raise ValueError("Valid options are: {}".format(self._options))

        show_image = False
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
            show_image = True
        ax.margins(AX_MARGIN)

        # Get the units for x- and y- axes
        x_unit = get_default_unit(var_x) if x_unit is None else x_unit
        y_unit = get_default_unit(var_y) if y_unit is None else y_unit

        x_unit_label, x_scale = get_unit_scale(x_unit)
        y_unit_label, y_scale = get_unit_scale(y_unit)

        x_sample, y_sample, density_color, idx_sample = sample_data(
            self._get_column(var_x),
            self._get_column(var_y),
            bins=bins_2d,
            sigma=sigma_2d,
            sample=sample)

        x_symmetric = False
        y_symmetric = False
        if var_x in ('x', 'xp'):
            x_symmetric = True
        if var_y in ('y', 'yp'):
            y_symmetric = True
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=MAX_LOCATOR,
                                                      symmetric=x_symmetric))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=MAX_LOCATOR,
                                                      symmetric=y_symmetric))

        if cloud_plot is False:
            cb = ax.scatter(x_sample*x_scale, y_sample*y_scale,
                            c=density_color, edgecolor='', s=ms, alpha=alpha, cmap='jet')
            cbar = plt.colorbar(cb, shrink=0.5)
            cbar.set_ticks(np.arange(0, 1.01, 0.2))
            cbar.ax.tick_params(labelsize=14)
        else:
            ax.scatter(x_sample * x_scale, y_sample * y_scale, alpha=alpha,
                       c=mc, edgecolor='', s=ms)

        ax.set_xlabel(get_label(var_x) + ' ' + x_unit_label,
                      fontsize=LABEL_FONT_SIZE, labelpad=LABEL_PAD)
        ax.set_ylabel(get_label(var_y) + ' ' + y_unit_label,
                      fontsize=LABEL_FONT_SIZE, labelpad=LABEL_PAD)
        ax.tick_params(labelsize=TICK_FONT_SIZE, pad=TICK_PAD)

        # set axis limits
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if show_parameters is True:
            # show parameters in the title for several plots
            if (var_x, var_y) == ('x', 'xp'):
                plt.title(r'$\varepsilon_x$ = %s $\mu$m' %
                          float("%.2g" % (self.params.emitx*1e6)),
                          fontsize=TITLE_FONT_SIZE, y=1.02)

            if (var_x, var_y) == ('y', 'yp'):
                plt.title(r'$\varepsilon_y$ = %s $\mu$m' %
                          float("%.2g" % (self.params.emity*1e6)),
                          fontsize=TITLE_FONT_SIZE, y=1.02)

            if (var_x, var_y) == ('t', 'p'):
                plt.title(r"$\sigma_t$ = %s " % float("%.2g" % (self.params.St*x_scale))
                          + x_unit_label.replace('(', '').replace(')', '')
                          + r", $\sigma_\delta$ = %s " % float("%.2g" % self.params.Sdelta)
                          + r", $Q$ = %s pC" % float("%.2g" % (self.params.charge*1e12)),
                          fontsize=TITLE_FONT_SIZE, y=1.02)
        else:
            plt.title(' ', fontsize=TITLE_FONT_SIZE, y=1.02)  # placeholder

        plt.tight_layout()

        if save_image is True:
            if not filename:
                filename = self.pfile.replace('.', '_') \
                         + '_' + var_x + '-' + var_y + '.png'
            else:
                filename = os.path.join(os.path.dirname(self.pfile), filename)
            plt.savefig(filename, dpi=dpi)
            print('%s saved!' % os.path.abspath(filename))
        else:
            plt.show()

    def _get_column(self, name):
        """Get the column data by name.

        :param name: string
            Name of the column data.
        """
        if name == 'p':
            return np.sqrt(self.data['px']**2 + self.data['py']**2 + self.data['pz']**2)

        if name in ('xp', 'yp'):
            return self.data[name[1] + name[0]] / self.data['pz']

        return self.data[name]


class AstraPhaseSpacePlot(PhaseSpacePlot):
    """Plot phase-spaces from ASTRA simulations."""
    def __init__(self, pfile, **kwargs):
        super().__init__(pfile, **kwargs)

    def _load_data(self):
        """Read data from file."""
        data, self.charge = parse_astra_phasespace(self.pfile)
        return data


class ImpacttPhaseSpacePlot(PhaseSpacePlot):
    """Plot phase-spaces from IMPACT-T simulations."""
    def __init__(self, pfile, charge, **kwargs):
        """"""
        super().__init__(pfile, **kwargs)
        self.charge = charge

    def _load_data(self):
        """Read data from file."""
        data = parse_impactt_phasespace(self.pfile)
        return data
