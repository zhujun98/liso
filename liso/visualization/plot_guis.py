#!/usr/bin/python
"""
Author: Jun Zhu
"""
import os
from abc import abstractmethod, ABCMeta

from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QAbstractItemView
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout
from PyQt5.QtWidgets import QDesktopWidget
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QListWidget
from PyQt5.QtWidgets import QGroupBox
from PyQt5.QtWidgets import QPushButton, QRadioButton
from PyQt5.QtWidgets import QSlider
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QAction, QActionGroup
from PyQt5.QtWidgets import QComboBox
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import QPen, QBrush, QColor

import pyqtgraph as pg

from ..data_processing import parse_line, parse_phasespace
from .vis_utils import *
from ..helpers import get_code


CUSTOM_PEN = [
    {'color': 'y', 'style': Qt.SolidLine},
    {'color': 'g', 'style': Qt.DashLine},
]
ICON_IMAGE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icons')


class PlotGUI(QMainWindow):
    """Abstract class of Plot GUI"""
    __metaclass__ = ABCMeta

    def __init__(self, parent=None):
        """Initialization."""
        super().__init__(parent=parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self._file_menu = self.menuBar().addMenu("File")
        self._style_menu = self.menuBar().addMenu("Style")
        self._setting_menu = self.menuBar().addMenu("Settings")

        self._data = None

        self._main_frame = QWidget()
        self._view = QWidget()
        self._control_panel = QWidget()
        self._set_main_frame()

        self._filepath_le = None
        self._code_cb = None
        self._graph = None
        self._plot = None
        self._set_view()

    def _set_menus(self):
        self._set_file_menu()
        self._set_style_menu()
        self._set_setting_menu()

    def _set_file_menu(self):
        """Set the 'File' menu."""
        choose_file_act = QAction("Load Data", self._file_menu)
        choose_file_act.triggered.connect(self._load_data)
        choose_file_act.setToolTip("Choose the particle file or the file con")

        quit_act = QAction("Quit", self._file_menu)
        quit_act.triggered.connect(self.close)
        quit_act.setToolTip("Close the application")

        self._add_actions(self._file_menu, (choose_file_act, quit_act))

    @abstractmethod
    def _set_style_menu(self):
        pass

    @abstractmethod
    def _set_setting_menu(self):
        """Set the 'settings' menu."""
        pass

    def _set_main_frame(self):
        """Set the layout of the main frame window."""
        layout = QHBoxLayout()
        layout.addWidget(self._view)
        layout.addWidget(self._control_panel)

        self._main_frame.setLayout(layout)
        self.setCentralWidget(self._main_frame)

    def _set_view(self):
        """Set the view window, in which the plot is displayed."""
        # A combo-box to choose the code
        self._code_cb = QComboBox(self._view)
        self._code_cb.addItems(['Astra', 'ImpactT', 'ImpactZ', 'Genesis'])
        self._code_cb.currentIndexChanged.connect(self._reset)

        # A text-box to show the current file (rootname) path
        self._filepath_le = QLineEdit(self._view)
        # signal returnPressed() does not emit after quiting
        self._filepath_le.returnPressed.connect(self._reload_data)

        # A button to reload the data
        synchronize_btn = QPushButton(self._view)
        synchronize_btn.setFixedSize(QSize(26, 26))
        synchronize_btn.setIcon(QIcon(os.path.join(ICON_IMAGE_FOLDER, 'synchronize.png')))
        synchronize_btn.setIconSize(QSize(26, 26))
        synchronize_btn.clicked.connect(self._reload_data)

        # A pyqtgraph.GraphicsLayoutWidget to show the plot
        self._graph = pg.GraphicsLayoutWidget(self._view)
        self._graph.setFixedSize(self._width, self._height)
        self._plot = self._graph.addPlot(title=" ")  # title is a placeholder

        layout = QGridLayout()
        layout.addWidget(synchronize_btn, 0, 0, 1, 1)
        layout.addWidget(self._code_cb, 0, 1, 1, 1)
        layout.addWidget(self._filepath_le, 0, 2, 1, 1)
        layout.addWidget(self._graph, 1, 0, 1, 3)

        self._view.setLayout(layout)

    @abstractmethod
    def _set_control_panel(self):
        """Set the control panel window."""
        pass

    @abstractmethod
    def _load_data(self):
        """Choose a file (rootname) and then Load data."""
        pass

    @abstractmethod
    def _reload_data(self):
        """Reload data from the chosen file(s)."""
        pass

    @abstractmethod
    def _update_plot(self):
        """Update the plots with the loaded data."""
        pass

    @staticmethod
    def _add_actions(widget, actions):
        """Add actions to a widget."""
        for action in actions:
            if action is None:
                widget.addSeparator()
            else:
                widget.addAction(action)

    @staticmethod
    def _on_radiobutton_toggled(widget):
        """Toggle a Widget with a radio button.

        :param widget: QWidget object
            Target widget.
        """
        if widget.isEnabled():
            widget.setDisabled(True)
        else:
            widget.setEnabled(True)

    def _reset(self):
        """Reset Widgets' data."""
        self._data = None
        self._filepath_le.setText('')
        self._plot.clear()


class PhaseSpacePlotGUI(PlotGUI):
    """GUI for phasespace plots."""
    def __init__(self, parent=None):
        """Initialization."""
        # TODO: make _width, _height, etc. system parameters,
        # Maximum number of particles to show.
        self._width = 480
        self._height = 480
        self._max_display_particle = 10000
        super().__init__(parent=parent)
        self._set_menus()
        self.setWindowTitle('Phase Space Plot')

        # Set the control panel
        self._xvar_list = None
        self._yvar_list = None
        self._markersize_sld = None
        self._sample_sld = None
        self._set_control_panel()

        self.show()

        self.setFixedSize(self.width(), self.height())
        # center the new window
        scr_size = QDesktopWidget().screenGeometry()
        x0 = int((scr_size.width() - self.frameSize().width()) / 2)
        y0 = int((scr_size.height() - self.frameSize().height()) / 2)
        self.move(x0, y0)

    def _set_setting_menu(self):
        pass
        # size1 = QAction("480x480 (default)", self._setting_menu)
        # size1.triggered.connect(lambda: self._set_graph_size(480, 480))
        #
        # size2 = QAction("600x600", self._setting_menu)
        # size2.triggered.connect(lambda: self._set_graph_size(600, 600))
        #
        # self._add_actions(self._setting_menu, (size1, size2))

    def _set_graph_size(self, w, h):
        self._graph.setFixedSize(w, h)

    def _set_style_menu(self):
        pass

    def _set_control_panel(self):
        self._control_panel.setFixedWidth(180)

        # =============================================================
        # Two lists which allows to select the variables for x- and y-
        # axes respectively.
        phasespace_options = ["x", "xp", "y", "yp", "dz", "t", "p", "delta"]

        var_lists_gp = QGroupBox()

        x_label = QLabel('x-axis')
        self._xvar_list = QListWidget()
        self._xvar_list.setFixedSize(60, 180)
        self._xvar_list.addItems(phasespace_options)
        self._xvar_list.itemClicked.connect(self._update_plot)

        y_label = QLabel('y-axis')
        self._yvar_list = QListWidget()
        self._yvar_list.setFixedSize(60, 180)
        self._yvar_list.addItems(phasespace_options)
        self._yvar_list.itemClicked.connect(self._update_plot)

        layout = QGridLayout()
        layout.addWidget(x_label, 0, 0, 1, 1)
        layout.addWidget(self._xvar_list, 1, 0, 1, 1)
        layout.addWidget(y_label, 0, 1, 1, 1)
        layout.addWidget(self._yvar_list, 1, 1, 1, 1)
        var_lists_gp.setLayout(layout)

        # =============================================================
        # A slider which adjusts the marker size and a radiobutton which
        # toggles the slider.
        markersize_gp = QGroupBox()

        markersize_rbtn = QRadioButton('Marker size')
        markersize_rbtn.setChecked(True)
        markersize_rbtn.toggled.connect(lambda: self._on_radiobutton_toggled(self._markersize_sld))

        self._markersize_sld = QSlider(Qt.Horizontal)
        self._markersize_sld.setTickPosition(QSlider.TicksBelow)
        self._markersize_sld.setRange(1, 15)
        self._markersize_sld.setValue(5)
        self._markersize_sld.setTickInterval(1)
        self._markersize_sld.setSingleStep(1)
        self._markersize_sld.valueChanged.connect(self._update_plot)

        vbox = QVBoxLayout()
        vbox.addWidget(markersize_rbtn)
        vbox.addWidget(self._markersize_sld)
        markersize_gp.setLayout(vbox)

        # =============================================================
        # A slider which adjusts the No. of particle shown on the screen
        # and a radiobutton which toggles the slider.
        sample_gp = QGroupBox()

        sample_rbtn = QRadioButton('No. of particles')
        sample_rbtn.setChecked(True)
        sample_rbtn.toggled.connect(lambda: self._on_radiobutton_toggled(self._sample_sld))

        self._sample_sld = QSlider(Qt.Horizontal)
        self._sample_sld.setTickPosition(QSlider.TicksBelow)
        self._set_sample_slider(self._max_display_particle)
        self._sample_sld.valueChanged.connect(self._update_plot)

        vbox = QVBoxLayout()
        vbox.addWidget(sample_rbtn)
        vbox.addWidget(self._sample_sld)
        sample_gp.setLayout(vbox)

        layout = QVBoxLayout()
        layout.addWidget(var_lists_gp)
        layout.addWidget(markersize_gp)
        layout.addWidget(sample_gp)
        layout.addStretch(1)

        self._control_panel.setLayout(layout)

    def _load_data(self):
        filepath, _ = QFileDialog.getOpenFileName(self, 'Open file', '/home')
        self._filepath_le.setText(filepath)
        self._reload_data()

    def _reload_data(self):
        filepath = self._filepath_le.text()
        if not filepath:
            return

        try:
            self._data, charge = parse_phasespace(get_code(self._code_cb.currentText()), filepath)
            # update the slider properties
            self._set_sample_slider(min(self._data.shape[0], self._max_display_particle))

            print("Loaded {} data from {}".format(self._code_cb.currentText(), filepath))
        except FileNotFoundError as e:
            print(e)
            return

        self._update_plot()

    def _update_plot(self):
        if not self._xvar_list.currentItem() or not self._yvar_list.currentItem():
            return

        if self._data is not None:
            x_var = self._xvar_list.currentItem().text()
            y_var = self._yvar_list.currentItem().text()

            x_unit = get_default_unit(x_var)
            y_unit = get_default_unit(y_var)

            x_unit_label, x_scale = get_unit_label_and_scale(x_unit)
            y_unit_label, y_scale = get_unit_label_and_scale(y_unit)

            x = get_phasespace_column_by_name(self._data, x_var)
            y = get_phasespace_column_by_name(self._data, y_var)
            x, y = fast_sample_data(x, y, n=self._sample_sld.value())
            self._plot.plot(x*x_scale, y*y_scale,
                            pen=None,
                            symbol='o',
                            symbolSize=self._markersize_sld.value(),
                            clear=True)

            self._plot.setLabel('left', get_html_label(y_var) + " " + y_unit_label)
            self._plot.setLabel('bottom', get_html_label(x_var) + " " + x_unit_label)

    def _set_sample_slider(self, v_max):
        """Set properties of a QSlider object.

        Those properties are allowed to change dynamically.

        :param v_max: int
            Maximum value.
        """
        self._sample_sld.setRange(0, v_max)
        self._sample_sld.setValue(v_max)
        self._sample_sld.setTickInterval(round(v_max/10))
        self._sample_sld.setSingleStep(round(v_max/10))

    def _reset(self):
        super()._reset()
        self._xvar_list.clearSelection()
        self._yvar_list.clearSelection()


class LinePlotGUI(PlotGUI):
    """GUI for line plots."""
    def __init__(self, parent=None):
        """Initialization."""
        self._width = 600
        self._height = 450
        super().__init__(parent=parent)
        self._set_menus()
        self.setWindowTitle('Beam Evolution Plot')

        self.legend = None

        # set the control panel
        self._var_list = None
        self._linewidth_sld = None
        self._set_control_panel()

        self.show()

        self.setFixedSize(self.width(), self.height())
        # center the new window
        scr_size = QDesktopWidget().screenGeometry()
        x0 = int((scr_size.width() - self.frameSize().width()) / 2)
        y0 = int((scr_size.height() - self.frameSize().height()) / 2)
        self.move(x0, y0)

    def _set_setting_menu(self):
        pass

    def _set_style_menu(self):
        pass

    def _set_control_panel(self):
        self._control_panel.setFixedWidth(180)

        # =============================================================
        # Two lists which allows to select the variables for x- and y-
        # axes respectively.
        varlist_gp = QGroupBox(self._control_panel)
        var_label = QLabel('Options', varlist_gp)
        var_label.setToolTip('Use "Ctrl + Click" to select more than one items or de-select item!')
        var_options = ['gamma', 'Sx', 'Sy', 'Sz', 'St', 'betax', 'betay', 'alphax', 'alphay',
                       'emitx', 'emity', 'emitz', 'SdE', 'emitx_tr', 'emity_tr']
        self._var_list = QListWidget(varlist_gp)
        self._var_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._var_list.setFixedSize(80, 200)
        self._var_list.addItems(var_options)
        self._var_list.itemSelectionChanged.connect(self._update_plot)

        vbox = QVBoxLayout()
        vbox.addWidget(var_label)
        vbox.addWidget(self._var_list)
        varlist_gp.setLayout(vbox)

        # =============================================================
        # A slider which adjusts the line width and a radiobutton which
        # toggles the slider.
        linewidth_gp = QGroupBox(self._control_panel)

        linewidth_rbtn = QRadioButton('Line width', linewidth_gp)
        linewidth_rbtn.setChecked(True)
        linewidth_rbtn.toggled.connect(lambda: self._on_radiobutton_toggled(self._linewidth_sld))

        self._linewidth_sld = QSlider(Qt.Horizontal, linewidth_gp)
        self._linewidth_sld.setTickPosition(QSlider.TicksBelow)
        self._linewidth_sld.setRange(1, 5)
        self._linewidth_sld.setValue(2)
        self._linewidth_sld.setTickInterval(1)
        self._linewidth_sld.setSingleStep(1)
        self._linewidth_sld.valueChanged.connect(self._update_plot)

        vbox = QVBoxLayout()
        vbox.addWidget(linewidth_rbtn)
        vbox.addWidget(self._linewidth_sld)
        linewidth_gp.setLayout(vbox)

        layout = QVBoxLayout()
        layout.addWidget(varlist_gp)
        layout.addWidget(linewidth_gp)
        layout.addStretch(1)

        self._control_panel.setLayout(layout)

    def _load_data(self):
        filepath, _ = QFileDialog.getOpenFileName(self, 'Open file', '/home')
        self._filepath_le.setText(filepath.split('.')[0])
        self._reload_data()

    def _reload_data(self):
        rootname = self._filepath_le.text()
        if not rootname:
            return

        try:
            self._data = parse_line(get_code(self._code_cb.currentText()), rootname)
            print("Loaded {} data from {}".format(self._code_cb.currentText(), rootname))
        except FileNotFoundError as e:
            print(e)
            return

        self._update_plot()

    def _update_plot(self):
        if not self._var_list.currentItem():
            return

        # Do not allow to select more than 2 items
        if self._var_list.selectedItems().__len__() > 2:
            self._var_list.currentItem().setSelected(False)
            return

        if self._data is not None:
            x_var = 'z'
            x = get_line_column_by_name(self._data, x_var)
            x_unit = get_default_unit(x_var)
            x_unit_label, x_scale = get_unit_label_and_scale(x_unit)

            lw = self._linewidth_sld.value()
            self._plot.clear()

            n_items = self._var_list.selectedItems().__len__()
            if n_items == 0:
                return
            if n_items > 1:
                self.legend = self._plot.addLegend()
            # TODO: remove legend if only there is only one line
            # else:
            #     if self.legend is not None:
            #         self.legend.scene().removeItem(self.legend)
            #     self.legend = None
            for i, item in enumerate(self._var_list.selectedItems()):
                y_var = item.text()
                y = get_line_column_by_name(self._data, y_var)
                y_unit = get_default_unit(y_var)
                y_unit_label, y_scale = get_unit_label_and_scale(y_unit)

                self._plot.plot(x*x_scale, y*y_scale,
                                name=get_html_label(y_var),
                                pen=pg.mkPen(CUSTOM_PEN[i]['color'],
                                width=lw,
                                style=CUSTOM_PEN[i]['style']))

            y_var_label = ", ".join([get_html_label(item.text())
                                     for item in self._var_list.selectedItems()])

            self._plot.setLabel('bottom', get_html_label(x_var) + " " + x_unit_label)
            self._plot.setLabel('left',  y_var_label + " " + y_unit_label)

    def _reset(self):
        super()._reset()
        self._filepath_le.setText('')
        self._var_list.clearSelection()
