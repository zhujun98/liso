#!/usr/bin/python
"""
Author: Jun Zhu
"""
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
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

        self.slider_lw = None

        self.main_frame = QWidget()

        self.fname_label = QLabel()
        self.view = pg.GraphicsLayoutWidget()
        self.psplot = self.view.addPlot(title=" ")  # title is a placeholder

        self.control_panel = QWidget()
        self.var_list = QListWidget(self)
        self.set_control_panel()

        self.set_main_frame()

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

    def set_main_frame(self):
        """"""
        self.view.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        layout = QGridLayout()

        layout.addWidget(self.fname_label, 0, 0, 1, 1)
        layout.addWidget(self.view, 1, 0, 1, 1)
        layout.addWidget(self.control_panel, 0, 1, 2, 1)

        self.main_frame.setLayout(layout)
        self.setCentralWidget(self.main_frame)

    def open_file(self, code):
        """Initialize a PhaseSpacePlot object."""
        filename, _ = QFileDialog.getOpenFileName(self, 'Open file', '/home')

        if not filename:
            return
        rootname = filename.split('.')[0]
        self.data = parse_line(code, rootname)

        self.fname_label.setText(rootname)
        self.update_plot()

    def set_control_panel(self):
        """"""
        self.control_panel.setFixedWidth(180)

        # =============================================================
        # Two lists which allows to select the variables for x- and y-
        # axes respectively.
        var_options = ['gamma', 'Sx', 'Sy', 'Sz', 'St',
                       'betax', 'betay', 'alphax', 'alphay',
                       'emitx', 'emity', 'emitz',
                       'SdE', 'emitx_tr', 'emity_tr']

        self.var_list.setFixedSize(80, 200)
        self.var_list.addItems(var_options)
        self.var_list.itemClicked.connect(self.update_plot)

        # =============================================================
        # A slider which adjusts the line width and a radiobutton which
        # toggles the slider.
        group_lw = QGroupBox(self.control_panel)
        radiobutton_lw = QRadioButton('Line width')
        radiobutton_lw.setChecked(True)
        radiobutton_lw.toggled.connect(lambda: self._on_radiobutton_toggled(self.slider_lw))
        self.slider_lw = QSlider(Qt.Horizontal)
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

        layout = QGridLayout()
        layout.addWidget(self.var_list, 0, 0, 1, 1)
        layout.addWidget(group_lw, 1, 0, 1, 1)
        layout.setRowStretch(2, 1)

        self.control_panel.setLayout(layout)

    def update_plot(self):
        """Update the phase-space plot."""
        if not self.var_list.currentItem():
            return

        if self.data is not None:
            var_x = 'z'
            var_y = self.var_list.currentItem().text().lower()
            x = get_line_column_by_name(self.data, var_x)
            y = get_line_column_by_name(self.data, var_y)

            x_unit = get_default_unit(var_x)
            y_unit = get_default_unit(var_y)

            x_unit_label, x_scale = get_unit_label_and_scale(x_unit)
            y_unit_label, y_scale = get_unit_label_and_scale(y_unit)

            width = self.slider_lw.value()
            self.psplot.plot(x*x_scale, y*y_scale, clear=True,
                             pen=pg.mkPen('y', width=width, style=Qt.SolidLine))

            self.psplot.setLabel('left', get_html_label(var_y) + " " + y_unit_label)
            self.psplot.setLabel('bottom', get_html_label(var_x) + " " + x_unit_label)

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

    def _set_sample_slider(self, v_max, *, tick_intervals=10):
        """Set properties of a QSlider object.

        Those properties are allowed to change dynamically.

        :param v_max: int
            Maximum value.
        :param tick_intervals:
            Number of tick intervals.
        """
        self.slider_sample.setRange(0, v_max)
        self.slider_sample.setValue(v_max)
        self.slider_sample.setTickInterval(round(v_max/tick_intervals))
        self.slider_sample.setSingleStep(round(v_max/tick_intervals))
