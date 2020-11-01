"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from ..proc import Phasespace, mesh_phasespace
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

    def plot(self, var_x, var_y, *,
             samples=20000,
             x_unit=None,
             y_unit=None,
             xlim=None,
             ylim=None,
             ms=2,
             mc='dodgerblue',
             alpha=1.0,
             show_parameters=True,
             show_current=False,
             ax=None):
        """Plot a given phasespace.

        :param string var_x: name of variable at x-axis (case insensitive).
        :param string var_y: name of variable at y-axis (case insensitive).
        :param int samples: number of data to be sampled.
        :param string x_unit: unit for x axis.
        :param string y_unit: unit for y axis.
        :param tuple xlim: range (x_min, x_max) of the x axis.
        :param tuple ylim: range (y_min, y_max) of the y axis.
        :param int ms: marker size.
        :param string mc: marker color.
        :param float alpha: alpha value (transparency). Default = 1.0.
        :param bool show_parameters: display beam parameters in the title.
            Default = True.
        :param bool show_current: show current plot using the y2 axis if the
            x axis is t or dt.
        :param matplotlib.axes.Axes/None ax: axis of the plot.
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

        if show_current and var_x in ('t', 'dt'):
            y2_unit = get_default_unit('i')
            y2_unit_label, y2_scale = get_unit_label_and_scale(y2_unit)

            ax2 = ax.twinx()
            ax2.margins(self._ax_margin)
            ax2.yaxis.set_major_locator(ticker.MaxNLocator(
                nbins=self._max_locator))
            ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

            t_centers = self._params.current_dist[0]
            if var_x == 'dt':
                # t_centers should not be modified inplace
                t_centers = t_centers - self._params.Ct
            ax2.plot(t_centers * x_scale,
                     self._params.current_dist[1] * y2_scale,
                     ls='--',
                     lw=2,
                     color='indigo')
            ax2.set_ylabel("$I$ " + y2_unit_label,
                           fontsize=self._label_fontsize,
                           labelpad=self._label_pad)
            ax2.tick_params(labelsize=self._tick_fontsize)

        # sample phasespace
        if samples >= len(self._data):
            x_sample, y_sample = self._data[var_x], self._data[var_y]
        else:
            # self._data[var] returns a pandas.Series
            x_sample = self._data[var_x].sample(samples).values
            y_sample = self._data[var_y].sample(samples).values

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

            elif var_x in ('t', 'dt') and var_y in ('p', 'delta'):
                ax.set_title(
                    r"$\sigma_t$ = %s " % float("%.2g" % (self._params.St * x_scale))
                    + x_unit_label.replace('(', '').replace(')', '')
                    + r", $\sigma_\delta$ = %s " % float("%.2g" % self._params.Sdelta)
                    + r", $Q$ = %s pC" % float("%.2g" % (self._params.charge * 1e12)),
                    fontsize=self._tick_fontsize, y=1.02)

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

    def imshow(self, var_x, var_y, *,
               x_bins=64,
               y_bins=64,
               x_range=None,
               y_range=None,
               x_unit=None,
               y_unit=None,
               cmap=None,
               ax=None,
               flip_origin=True):
        """Plot the phasespace as a pseudo image.

        :param string var_x: name of variable at x-axis (case insensitive).
        :param string var_y: name of variable at y-axis (case insensitive).
        :param int x_bins: number of bins in x.
        :param int y_bins: number of bins in y.
        :param tuple x_range: binning range (x_min, x_max) in x,
            and (x_max - x_min) / x_bins should be the pixel size in x.
        :param tuple y_range: binning range (y_min, y_max) in y,
            and (y_max - y_min) / y_bins should be the pixel size in y.
        :param string x_unit: unit for x axis.
        :param string y_unit: unit for y axis.
        :param matplotlib.axes.Axes/None ax: axis of the plot.
        :param bool flip_origin: True for flipping the y_axis.
        """
        var_x = var_x.lower()
        var_y = var_y.lower()

        x_label, x_unit_label, x_scale = self._get_label_and_scale(
            var_x, x_unit)
        y_label, y_unit_label, y_scale = self._get_label_and_scale(
            var_y, y_unit)

        if ax is None:
            _, ax = plt.subplots()

        intensity, xc, yc = mesh_phasespace(
            self._data[var_x] * x_scale, self._data[var_y] * y_scale,
            x_bins=x_bins,
            y_bins=y_bins,
            x_range=x_range,
            y_range=y_range)

        if cmap is None:
            cmap = 'viridis'

        ax.imshow(intensity,
                  aspect='auto',
                  cmap=cmap,
                  extent=[xc.min(), xc.max(), yc.min(), yc.max()],
                  origin='lower' if flip_origin else 'upper')

        self._set_labels_and_tick(
            ax, (x_label, y_label), (x_unit_label, y_unit_label))

        return ax
