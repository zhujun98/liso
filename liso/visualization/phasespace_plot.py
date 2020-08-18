"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from ..data_processing import analyze_beam, tailor_beam
from .vis_utils import (
    fast_sample_data, get_default_unit, get_label,
    get_phasespace_column_by_name, get_unit_label_and_scale, sample_data
)


class PhasespacePlot(object):
    """Plot the beam phase-space."""

    _options = ['x', 'y', 'dz', 'xp', 'yp', 't', 'p', 'delta']

    def __init__(self, data, charge, *,
                 halo=0.0,
                 tail=0.0,
                 rotation=0.0,
                 figsize=(8, 6),
                 label_fontsize=18,
                 tick_fontsize=14,
                 legend_fontsize=16,
                 label_pad=8,
                 tick_pad=8,
                 max_locator=6,
                 ax_margin=0.05,
                 **kwargs):
        """Initialization.

        :param Pandas.DataFrame data: phasespace data.
        :param float charge: bunch charge.
        :param halo: float
            Percentage of particles to be removed based on their
            transverse distance to the bunch centroid. Applied
            before tail cutting.
        :param tail: float
            Percentage of particles to be removed in the tail.
        :param rotation: float
            Angle of the rotation in rad.
        """
        self._data, self._charge = data, charge

        n0 = len(self._data)
        self._data = tailor_beam(self._data,
                                 tail=tail, halo=halo, rotation=rotation)

        self._charge *= len(self._data) / n0

        self.params = analyze_beam(self._data, self._charge, **kwargs)

        self._figsize = figsize

        self._label_fontsize = label_fontsize
        self._tick_fontsize = tick_fontsize
        self._legend_fontsize = legend_fontsize

        self._label_pad = label_pad
        self._tick_pad = tick_pad

        self._max_locator = max_locator

        self._ax_margin = ax_margin

    def cloud(self, var_x, var_y, **kwargs):
        self._plot(var_x, var_y, **kwargs)

    def scatter(self, var_x, var_y, **kwargs):
        self._plot(var_x, var_y, cloud_plot=False, **kwargs)

    def _plot(self, var_x, var_y, *,
              samples=20000,
              ax=None,
              x_unit=None,
              y_unit=None,
              y1_unit=None,
              xlim=None,
              ylim=None,
              cloud_plot=True,
              ms=2,
              mc='dodgerblue',
              alpha=1.0,
              bins_2d=500,
              sigma_2d=5,
              show_parameters=True):
        """Show a phase-space on screen or in a file.

        :param string var_x: name of variable at x-axis (case insensitive).
        :param string var_y: name of variable at y-axis (case insensitive).
        :param int samples: number of data to be sampled.
        :param string x_unit: unit for x axis.
        :param string y_unit: unit for y axis.
        :param string y1_unit: unit for y2 axis.
        :param tuple xlim: range (x_min, x_max) of the x axis.
        :param tuple ylim: range (y_min, y_max) of the y axis.
        :param bool cloud_plot: true for colorful density plot.
        :param int ms: marker size for scatter plots.
        :param string mc: color of markers for non-density plot.
        :param float alpha: alpha value (transparency). Default = 1.0.
        :param int/[int, int] bins_2d: number of bins used in
            numpy.histogram2d.
        :param float sigma_2d: standard deviation of Gaussian kernel
            of the Gaussian filter.
        :param bool show_parameters: display beam parameters in the title.
            Default = True.
        """
        var_x = var_x.lower()
        var_y = var_y.lower()
        if var_x not in self._options or var_y not in self._options:
            raise ValueError(f"Valid options are: {self._options}")

        # get the units for x- and y- axes
        x_unit = get_default_unit(var_x) if x_unit is None else x_unit
        y_unit = get_default_unit(var_y) if y_unit is None else y_unit

        x_unit_label, x_scale = get_unit_label_and_scale(x_unit)
        y_unit_label, y_scale = get_unit_label_and_scale(y_unit)

        if ax is None:
            fig = plt.figure(figsize=self._figsize, tight_layout=True)
            ax = fig.add_subplot(111)
        else:
            fig = None

        ax.margins(self._ax_margin)

        x_symmetric = False
        y_symmetric = False
        if var_x in ('x', 'xp'):
            x_symmetric = True
        if var_y in ('y', 'yp'):
            y_symmetric = True
        ax.xaxis.set_major_locator(ticker.MaxNLocator(
            nbins=self._max_locator, symmetric=x_symmetric))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(
            nbins=self._max_locator, symmetric=y_symmetric))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

        if cloud_plot is True:
            x_sample, y_sample, density_color = sample_data(
                get_phasespace_column_by_name(self._data, var_x),
                get_phasespace_column_by_name(self._data, var_y),
                n=samples,
                bins=bins_2d,
                sigma=sigma_2d)

            cb = ax.scatter(x_sample*x_scale, y_sample*y_scale,
                            c=density_color,
                            s=ms,
                            alpha=alpha,
                            cmap='jet')

            if (var_x, var_y) == ('t', 'p'):
                y1_unit = get_default_unit('i') if y1_unit is None else y1_unit
                y1_unit_label, y1_scale = get_unit_label_and_scale(y1_unit)

                ax1 = ax.twinx()
                ax1.margins(self._ax_margin)
                ax1.yaxis.set_major_locator(ticker.MaxNLocator(
                    nbins=self._max_locator))
                ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

                ax1.plot(self.params.current_dist[0] * x_scale,
                         self.params.current_dist[1] * y1_scale,
                         ls='--',
                         lw=2,
                         color='indigo')
                ax1.set_ylabel("$I$ " + y1_unit_label,
                               fontsize=self._label_fontsize,
                               labelpad=self._label_pad)
                ax1.tick_params(labelsize=self._tick_fontsize)

                cbaxes = fig.add_axes([0.75, 0.07, 0.2, 0.02])
                cbar = plt.colorbar(cb, orientation='horizontal', cax=cbaxes)
            else:
                cbar = plt.colorbar(cb, shrink=0.5)

            cbar.set_ticks(np.arange(0, 1.01, 0.2))
            cbar.ax.tick_params(labelsize=14)

        else:
            x_sample, y_sample = fast_sample_data(
                get_phasespace_column_by_name(self._data, var_x),
                get_phasespace_column_by_name(self._data, var_y),
                n=samples)

            ax.scatter(x_sample * x_scale, y_sample * y_scale,
                       alpha=alpha, c=mc, s=ms)

        ax.set_xlabel(get_label(var_x) + ' ' + x_unit_label,
                      fontsize=self._label_fontsize, labelpad=self._label_pad)
        ax.set_ylabel(get_label(var_y) + ' ' + y_unit_label,
                      fontsize=self._label_fontsize, labelpad=self._label_pad)
        ax.tick_params(labelsize=self._tick_fontsize, pad=self._tick_pad)

        # set axis limits
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_title(' ', fontsize=self._tick_fontsize, y=1.02)  # placeholder
        if show_parameters is True:
            # show parameters in the title for several plots
            if (var_x, var_y) == ('x', 'xp'):
                ax.set_title(
                    r'$\varepsilon_x$ = %s $\mu$m' %
                    float("%.2g" % (self.params.emitx*1e6)),
                    fontsize=self._tick_fontsize, y=1.02)

            elif (var_x, var_y) == ('y', 'yp'):
                ax.set_title(
                    r'$\varepsilon_y$ = %s $\mu$m'
                    % float("%.2g" % (self.params.emity*1e6)),
                    fontsize=self._tick_fontsize, y=1.02)

            elif var_x == 't' and (var_y == 'p' or var_y == 'delta'):
                ax.set_title(
                    r"$\sigma_t$ = %s " % float("%.2g" % (self.params.St*x_scale))
                    + x_unit_label.replace('(', '').replace(')', '')
                    + r", $\sigma_\delta$ = %s " % float("%.2g" % self.params.Sdelta)
                    + r", $Q$ = %s pC" % float("%.2g" % (self.params.charge*1e12)),
                    fontsize=self._tick_fontsize, y=1.02)

        return ax
