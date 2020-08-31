"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from ..data_processing import (
    density_phasespace, Phasespace, pixel_phasespace, sample_phasespace
)
from .vis_utils import (
    get_default_unit, get_label, get_unit_label_and_scale
)


class PhasespacePlot(object):
    """Plot the beam phase-space."""

    def __init__(self, data, *,
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

        :param Phasespace data: phasespace data.
        :param float halo: Percentage of particles to be removed based
            on their transverse distance to the bunch centroid. Applied
            before tail cutting.
        :param float tail: Percentage of particles to be removed in the tail.
        :param float rotation: Angle of the rotation in rad.
        """
        if not isinstance(data, Phasespace):
            raise TypeError("data must be a Phasespace object!")
        self._data = data

        n0 = len(self._data)

        self._data.rotate(rotation)
        self._data.cut_tail(tail)
        self._data.cut_halo(halo)

        self._data.charge *= len(self._data) / n0

        self._params = self._data.analyze(**kwargs)

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

        x_label, x_unit_label, x_scale = self._get_label_and_scale(
            var_x, x_unit)
        y_label, y_unit_label, y_scale = self._get_label_and_scale(
            var_y, y_unit)

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
            density, x_sample, y_sample = density_phasespace(
                self._data[var_x],
                self._data[var_y],
                n=samples,
                n_bins=bins_2d,
                sigma=sigma_2d)

            cb = ax.scatter(x_sample*x_scale, y_sample*y_scale,
                            c=density,
                            s=ms,
                            alpha=alpha,
                            cmap='jet')

            if (var_x, var_y) == ('dt', 'p'):
                y1_unit = get_default_unit('i') if y1_unit is None else y1_unit
                y1_unit_label, y1_scale = get_unit_label_and_scale(y1_unit)

                ax1 = ax.twinx()
                ax1.margins(self._ax_margin)
                ax1.yaxis.set_major_locator(ticker.MaxNLocator(
                    nbins=self._max_locator))
                ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

                ax1.plot(self._params.current_dist[0] * x_scale,
                         self._params.current_dist[1] * y1_scale,
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
            x_sample, y_sample = sample_phasespace(
                self._data[var_x], self._data[var_y], n=samples)

            ax.scatter(x_sample * x_scale, y_sample * y_scale,
                       alpha=alpha, c=mc, s=ms)

        self._set_labels_and_tick(
            ax, (x_label, y_label), (x_unit_label, y_unit_label))

        # set axis limits
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_title(' ', fontsize=self._tick_fontsize, y=1.02)  # placeholder
        if show_parameters is True:
            # show parameters in the title for several plots
            if (var_x, var_y) == ('x', 'xp'):
                ax.set_title(
                    r'$\varepsilon_x$ = %s $\mu$m' %
                    float("%.2g" % (self._params.emitx*1e6)),
                    fontsize=self._tick_fontsize, y=1.02)

            elif (var_x, var_y) == ('y', 'yp'):
                ax.set_title(
                    r'$\varepsilon_y$ = %s $\mu$m'
                    % float("%.2g" % (self._params.emity*1e6)),
                    fontsize=self._tick_fontsize, y=1.02)

            elif var_x == 'dt' and (var_y == 'p' or var_y == 'delta'):
                ax.set_title(
                    r"$\sigma_t$ = %s " % float("%.2g" % (self._params.St*x_scale))
                    + x_unit_label.replace('(', '').replace(')', '')
                    + r", $\sigma_\delta$ = %s " % float("%.2g" % self._params.Sdelta)
                    + r", $Q$ = %s pC" % float("%.2g" % (self._params.charge*1e12)),
                    fontsize=self._tick_fontsize, y=1.02)

        return ax

    def current(self, n_bins=128, *,
                ax=None, x_unit=None, y_unit=None, xlim=None, ylim=None):
        """Plot the current profile.

        :param int n_bins: number of bins used in histogram.
        """
        var_x, var_y = 'dt', 'i'
        x_label, x_unit_label, x_scale = self._get_label_and_scale(var_x, x_unit)
        y_label, y_unit_label, y_scale = self._get_label_and_scale(var_y, y_unit)

        t = self._data[var_x]
        hist, edges = np.histogram(t, bins=n_bins)
        centers = (edges[1:] + edges[:-1]) / 2.

        if ax is None:
            fig = plt.figure(figsize=self._figsize, tight_layout=True)
            ax = fig.add_subplot(111)

        self._set_labels_and_tick(
            ax, (x_label, y_label), (x_unit_label, y_unit_label))

        ax.plot(centers * x_scale, hist * y_scale)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        return ax

    def _get_label_and_scale(self, var, unit):
        unit = get_default_unit(var) if unit is None else unit
        unit_label, scale = get_unit_label_and_scale(unit)
        return get_label(var), unit_label, scale

    def _set_labels_and_tick(self, ax, labels, unit_labels):
        ax.set_xlabel(labels[0] + ' ' + unit_labels[0],
                      fontsize=self._label_fontsize,
                      labelpad=self._label_pad)
        ax.set_ylabel(labels[1] + ' ' + unit_labels[1],
                      fontsize=self._label_fontsize,
                      labelpad=self._label_pad)
        ax.tick_params(labelsize=self._tick_fontsize, pad=self._tick_pad)

    @classmethod
    def imshow(cls, x, y, *, ax=None, cmap=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots()

        i, xc, yc = pixel_phasespace(y, x, **kwargs)

        if cmap is None:
            cmap = 'viridis'

        ax.imshow(np.flip(i, axis=0),
                  aspect='auto',
                  cmap=cmap,
                  extent=[yc.min(), yc.max(), xc.min(), xc.max()])
