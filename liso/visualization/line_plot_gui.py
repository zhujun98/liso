#!/usr/bin/python
"""
Author: Jun Zhu
"""
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QAbstractItemView
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout
from PyQt5.QtWidgets import QDesktopWidget
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QListWidget
from PyQt5.QtWidgets import QGroupBox
from PyQt5.QtWidgets import QRadioButton
from PyQt5.QtWidgets import QSlider
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QAction
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPen, QBrush, QColor

import pyqtgraph as pg

from ..data_processing import parse_line
from .vis_utils import *


WINDOW_WIDTH = 600
WINDOW_HEIGHT = 450
CUSTOM_PEN = [
    {'color': 'y', 'style': Qt.SolidLine},
    {'color': 'g', 'style': Qt.DashLine},
]


class LinePlotGUI(QMainWindow):
    """"""
    def __init__(self, parent=None):
        """Initialization."""
        super().__init__(parent=parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('Beam Evolution Plot')

        # Maximum number of particles to show.
        self._max_display_particle = 10000

        self.file_menu = self.menuBar().addMenu("File")
        self.set_file_menu()
        self.style_menu = self.menuBar().addMenu("Style")
        self.set_style_menu()
        self.setting_menu = self.menuBar().addMenu("Settings")
        self.set_setting_menu()

        self.code = "a"
        self.data = None

        self.main_frame = QWidget()
        self.view = QWidget()
        self.control_panel = QWidget()
        self.set_main_frame()

        self.fname_label = None
        self.psplot = None
        self.legend = None
        self.set_view()

        self.var_list = None
        self.slider_lw = None
        self.set_control_panel()

        self.show()

        self.setFixedSize(self.width(), self.height())
        # center the new window
        scr_size = QDesktopWidget().screenGeometry()
        x0 = int((scr_size.width() - self.frameSize().width()) / 2)
        y0 = int((scr_size.height() - self.frameSize().height()) / 2)
        self.move(x0, y0)

    def set_file_menu(self):
        """Set file menu."""
        load_astra_file = QAction("Open ASTRA File", self)
        load_astra_file.triggered.connect(lambda: self.open_file("a"))
        load_astra_file.setToolTip("Open particle file")

        load_impactt_file = QAction("Open IMPACT-T File", self)
        load_impactt_file.triggered.connect(lambda: self.open_file("t"))
        load_impactt_file.setToolTip("Open particle file")

        load_impactz_file = QAction("Open IMPACT-Z File", self)
        load_impactz_file.triggered.connect(lambda: self.open_file("z"))
        load_impactz_file.setToolTip("Open particle file")

        load_genesis_file = QAction("Open GENESIS File", self)
        load_genesis_file.triggered.connect(lambda: self.open_file("g"))
        load_genesis_file.setToolTip("Open particle file")

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        quit_action.setToolTip("Close the application")

        self._add_actions(self.file_menu, (load_astra_file,
                                           load_impactt_file,
                                           load_impactz_file,
                                           load_genesis_file,
                                           quit_action))

    def set_setting_menu(self):
        """Set settings menu."""
        pass

    def set_style_menu(self):
        pass

    def open_file(self, code):
        """Initialize a PhaseSpacePlot object."""
        filename, _ = QFileDialog.getOpenFileName(self, 'Open file', '/home')

        if not filename:
            return
        rootname = filename.split('.')[0]
        self.data = parse_line(code, rootname)

        self.fname_label.setText(rootname)
        self.update_plot()

    def set_main_frame(self):
        """Set the layout of the main window."""
        layout = QHBoxLayout()
        layout.addWidget(self.view)
        layout.addWidget(self.control_panel)

        self.main_frame.setLayout(layout)
        self.setCentralWidget(self.main_frame)

    def set_view(self):
        """"""
        # QLabel to show the current rootname
        self.fname_label = QLabel(self.view)

        # pyqtgraph.GraphicsLayoutWidget to show the plot
        graph = pg.GraphicsLayoutWidget(self.view)
        graph.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.psplot = graph.addPlot(title=" ")  # title is a placeholder

        layout = QVBoxLayout()
        layout.addWidget(self.fname_label)
        layout.addWidget(graph)

        self.view.setLayout(layout)

    def set_control_panel(self):
        """"""
        self.control_panel.setFixedWidth(180)

        # =============================================================
        # Two lists which allows to select the variables for x- and y-
        # axes respectively.
        group_vl = QGroupBox(self.control_panel)
        var_label = QLabel('Options', group_vl)
        var_label.setToolTip(
            'Use "Ctrl + Click" to select more than one items or de-select item!')
        var_options = ['gamma', 'Sx', 'Sy', 'Sz', 'St',
                       'betax', 'betay', 'alphax', 'alphay',
                       'emitx', 'emity', 'emitz',
                       'SdE', 'emitx_tr', 'emity_tr']
        self.var_list = QListWidget(group_vl)
        self.var_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.var_list.setFixedSize(80, 200)
        self.var_list.addItems(var_options)
        self.var_list.itemSelectionChanged.connect(self.update_plot)

        vbox = QVBoxLayout()
        vbox.addWidget(var_label)
        vbox.addWidget(self.var_list)
        group_vl.setLayout(vbox)

        # =============================================================
        # A slider which adjusts the line width and a radiobutton which
        # toggles the slider.
        group_lw = QGroupBox(self.control_panel)

        radiobutton_lw = QRadioButton('Line width', group_lw)
        radiobutton_lw.setChecked(True)
        radiobutton_lw.toggled.connect(lambda: self._on_radiobutton_toggled(self.slider_lw))

        self.slider_lw = QSlider(Qt.Horizontal, group_lw)
        self.slider_lw.setTickPosition(QSlider.TicksBelow)
        self.slider_lw.setRange(1, 5)
        self.slider_lw.setValue(2)
        self.slider_lw.setTickInterval(1)
        self.slider_lw.setSingleStep(1)
        self.slider_lw.valueChanged.connect(self.update_plot)

        vbox = QVBoxLayout()
        vbox.addWidget(radiobutton_lw)
        vbox.addWidget(self.slider_lw)
        group_lw.setLayout(vbox)

        layout = QVBoxLayout()
        layout.addWidget(group_vl)
        layout.addWidget(group_lw)
        layout.addStretch(1)

        self.control_panel.setLayout(layout)

    def update_plot(self):
        """Update the phase-space plot."""
        if not self.var_list.currentItem():
            return

        # Do not allow to select more than 2 items
        if self.var_list.selectedItems().__len__() > 2:
            self.var_list.currentItem().setSelected(False)
            return

        if self.data is not None:
            var_x = 'z'
            x = get_line_column_by_name(self.data, var_x)
            x_unit = get_default_unit(var_x)
            x_unit_label, x_scale = get_unit_label_and_scale(x_unit)

            lw = self.slider_lw.value()
            self.psplot.clear()

            n_items = self.var_list.selectedItems().__len__()
            if n_items == 0:
                return
            if n_items > 1:
                self.legend = self.psplot.addLegend()
            # TODO: remove legend if only there is only one line
            # else:
            #     if self.legend is not None:
            #         self.legend.scene().removeItem(self.legend)
            #     self.legend = None
            for i, item in enumerate(self.var_list.selectedItems()):
                var_y = item.text()
                y = get_line_column_by_name(self.data, var_y)
                y_unit = get_default_unit(var_y)
                y_unit_label, y_scale = get_unit_label_and_scale(y_unit)

                self.psplot.plot(x*x_scale, y*y_scale, name=get_html_label(var_y),
                                 pen=pg.mkPen(CUSTOM_PEN[i]['color'],
                                              width=lw,
                                              style=CUSTOM_PEN[i]['style']))
            var_y_label = ", ".join([get_html_label(item.text())
                                     for item in self.var_list.selectedItems()])

            self.psplot.setLabel('bottom', get_html_label(var_x) + " " + x_unit_label)
            self.psplot.setLabel('left',  var_y_label + " " + y_unit_label)

    def _add_actions(self, target, actions):
        """"""
        for action in actions:
            if action is None:
                target.addSeparator()
            else:
                target.addAction(action)

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
