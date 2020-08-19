"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from .vis_utils import *


class LinePlot:
    """LinePlot class."""

    _options = ['gamma', 'sx', 'sy', 'sz', 'st',
                'betax', 'betay', 'alphax', 'alphay',
                'emitx', 'emity', 'emitz', 'sde', 'emitx_tr', 'emity_tr']

    def __init__(self, data, *,
                 figsize=(6, 4),
                 label_fontsize=18,
                 tick_fontsize=14,
                 legend_fontsize=16,
                 label_pad=8,
                 tick_pad=8,
                 max_locator=6,
                 ax_margin=0.05):
        """Initialization.

        :param Pandas.DataFrame data: line data.
        """
        self._data = data

        self._figsize = figsize

        self._label_fontsize = label_fontsize
        self._tick_fontsize = tick_fontsize
        self._legend_fontsize = legend_fontsize

        self._label_pad = label_pad
        self._tick_pad = tick_pad

        self._max_locator = max_locator

        self._ax_margin = ax_margin

    def plot(self, var1, var2=None, *,
             ax=None, x_unit=None, y_unit=None, xlim=None, ylim=None):
        """Plot parameters' evolution along the beamline.

        :param string var1: name of 1st variable (case insensitive).
        :param string var2: name of 2nd variable
            (optional and case insensitive).
        :param Axes ax: the axes to plot.
        :param string x_unit: units of x axis.
        :param string y_unit: units of y axis.
        :param tuple xlim: range (x_min, x_max) of the x axis.
        :param tuple ylim: range (y_min, y_max) of the y axis.
        """
        var1 = var1.lower()
        var2 = var2.lower() if var2 is not None else var2
        if var1 not in self._options or \
                (var2 is not None and var2 not in self._options):
            raise ValueError(f"Valid options are: {self._options}")

        x_unit = get_default_unit('z') if x_unit is None else x_unit
        # var2 should have the same y_unit
        y_unit = get_default_unit(var1) if y_unit is None else y_unit

        x_unit_label, x_scale = get_unit_label_and_scale(x_unit)
        y_unit_label, y_scale = get_unit_label_and_scale(y_unit)

        if ax is None:
            fig = plt.figure(figsize=self._figsize, tight_layout=True)
            ax = fig.add_subplot(111)

        ax.margins(self._ax_margin)

        ax.xaxis.set_major_locator(ticker.MaxNLocator(self._max_locator))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(self._max_locator))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

        colors = ['dodgerblue', 'firebrick']
        styles = ['-', '--']
        vars = [var1, var2]
        for i, var in enumerate(vars):
            if var is not None:
                x = get_line_column_by_name(self._data, 'z')
                y = get_line_column_by_name(self._data, var)
                ax.plot(x*x_scale, y*y_scale,
                        c=colors[i], ls=styles[i], lw=2, label=get_label(var))

        ax.set_xlabel("$z$ " + x_unit_label,
                      fontsize=self._label_fontsize, labelpad=self._label_pad)
        ax.set_ylabel("$,$ ".join([get_label(var) for var in vars
                                   if var is not None])
                      + " " + y_unit_label,
                      fontsize=self._label_fontsize, labelpad=self._label_pad)

        ax.tick_params(labelsize=self._tick_fontsize, pad=self._tick_pad)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if var2 is not None:
            ax.legend(loc=0, fontsize=self._legend_fontsize)

        return ax
