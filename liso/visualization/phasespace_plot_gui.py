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

import pyqtgraph as pg

from ..data_processing import parse_phasespace
from .vis_utils import *


WINDOW_WIDTH = 450
WINDOW_HEIGHT = 450


class PhaseSpacePlotGUI(QMainWindow):
    """GUI for phasespace plot."""
    def __init__(self, parent=None):
        """Initialization."""
        super().__init__(parent=parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('Phase Space Plot')

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
        self.view = QWidget(self.main_frame)
        self.control_panel = QWidget(self.main_frame)
        self.set_main_frame()

        # Set the graph window
        self.fname_label = None
        self.psplot = None
        self.set_view()

        # Set the control panel
        self.xlist = None
        self.ylist = None
        self.slider_ms = None
        self.slider_sample = None
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
        self.data, charge = parse_phasespace(code, filename)
        if charge is None:
            charge = 0.0

        # update the slider properties
        self._set_sample_slider(min(self.data.shape[0], self._max_display_particle))
        QApplication.processEvents()

        self.fname_label.setText(filename)
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
        # QLabel to show the current filename
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
        phasespace_options = ["x", "xp", "y", "yp", "dz", "t", "p", "delta"]

        group_lst = QGroupBox(self.control_panel)
        x_label = QLabel('x-axis', group_lst)
        self.xlist = QListWidget(group_lst)
        self.xlist.setFixedSize(60, 180)
        self.xlist.addItems(phasespace_options)
        self.xlist.itemClicked.connect(self.update_plot)

        y_label = QLabel('y-axis', group_lst)
        self.ylist = QListWidget(group_lst)
        self.ylist.setFixedSize(60, 180)
        self.ylist.addItems(phasespace_options)
        self.ylist.itemClicked.connect(self.update_plot)

        layout = QGridLayout()
        layout.addWidget(x_label, 0, 0, 1, 1)
        layout.addWidget(self.xlist, 1, 0, 1, 1)
        layout.addWidget(y_label, 0, 1, 1, 1)
        layout.addWidget(self.ylist, 1, 1, 1, 1)
        group_lst.setLayout(layout)

        # =============================================================
        # A slider which adjusts the marker size and a radiobutton which
        # toggles the slider.
        group_ms = QGroupBox(self.control_panel)

        radiobutton_ms = QRadioButton('Marker size', group_ms)
        radiobutton_ms.setChecked(True)
        radiobutton_ms.toggled.connect(lambda: self._on_radiobutton_toggled(self.slider_ms))

        self.slider_ms = QSlider(Qt.Horizontal, group_ms)
        self.slider_ms.setTickPosition(QSlider.TicksBelow)
        self.slider_ms.setRange(1, 15)
        self.slider_ms.setValue(5)
        self.slider_ms.setTickInterval(1)
        self.slider_ms.setSingleStep(1)
        self.slider_ms.valueChanged.connect(self.update_plot)

        vbox = QVBoxLayout()
        vbox.addWidget(radiobutton_ms)
        vbox.addWidget(self.slider_ms)
        group_ms.setLayout(vbox)

        # =============================================================
        # A slider which adjusts the No. of particle shown on the screen
        # and a radiobutton which toggles the slider.
        group_sample = QGroupBox(self.control_panel)

        radiobutton_sample = QRadioButton('No. of particles', group_sample)
        radiobutton_sample.setChecked(True)
        radiobutton_sample.toggled.connect(lambda: self._on_radiobutton_toggled(self.slider_sample))

        self.slider_sample = QSlider(Qt.Horizontal, group_sample)
        self.slider_sample.setTickPosition(QSlider.TicksBelow)
        self._set_sample_slider(self._max_display_particle)
        self.slider_sample.valueChanged.connect(self.update_plot)

        vbox = QVBoxLayout()
        vbox.addWidget(radiobutton_sample)
        vbox.addWidget(self.slider_sample)
        group_sample.setLayout(vbox)

        layout = QVBoxLayout()
        layout.addWidget(group_lst)
        layout.addWidget(group_ms)
        layout.addWidget(group_sample)
        layout.addStretch(1)

        self.control_panel.setLayout(layout)

    def update_plot(self):
        """Update the phase-space plot."""
        if not self.xlist.currentItem() or not self.ylist.currentItem():
            return

        if self.data is not None:
            var_x = self.xlist.currentItem().text()
            var_y = self.ylist.currentItem().text()

            x_unit = get_default_unit(var_x)
            y_unit = get_default_unit(var_y)

            x_unit_label, x_scale = get_unit_label_and_scale(x_unit)
            y_unit_label, y_scale = get_unit_label_and_scale(y_unit)

            x = get_phasespace_column_by_name(self.data, var_x)
            y = get_phasespace_column_by_name(self.data, var_y)
            x, y = fast_sample_data(x, y, n=self.slider_sample.value())
            self.psplot.plot(x*x_scale, y*y_scale,
                             pen=None,
                             symbol='o',
                             symbolSize=self.slider_ms.value(),
                             clear=True)

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
