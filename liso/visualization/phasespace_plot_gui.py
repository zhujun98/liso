#!/usr/bin/python
"""
Author: Jun Zhu
"""
import numpy as np

from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout
from PyQt5.QtWidgets import QDesktopWidget
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QComboBox
from PyQt5.QtWidgets import QListWidget
from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtWidgets import QGroupBox
from PyQt5.QtWidgets import QRadioButton
from PyQt5.QtWidgets import QSlider
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QAction
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtWidgets import qApp
from PyQt5.QtGui import QIcon

from .. import pyqtgraph as pg

from ..data_processing import parse_phasespace
from ..data_processing import analyze_beam
from .vis_utils import *


WINDOW_WIDTH = 600
WINDOW_HEIGHT = 450


class PhaseSpacePlotGUI(QMainWindow):
    """"""
    def __init__(self, parent=None):
        """Initialization."""
        super().__init__(parent=parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('Phase Space Plot')

        self.file_menu = self.menuBar().addMenu("&File")
        self.set_file_menu()

        self.code = "a"
        self.data = None

        self.main_frame = QWidget()

        self.fname_label = QLabel()
        self.view = pg.GraphicsLayoutWidget()
        self.psplot = self.view.addPlot()

        self.control_panel = QWidget()
        self.xlist = QListWidget(self)
        self.ylist = QListWidget(self)
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
        """Create menu bar."""
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
        # filename, _ = QFileDialog.getOpenFileName(self, 'Open file', '/home')

        # For test
        if code == "a":
            filename = "/home/jun/Projects/LISO/examples/astra_basic/injector.0400.001"
        elif code == "t":
            filename = "/home/jun/Projects/LISO/examples/impactt_basic/fort.107"
        else:
            return

        if not filename:
            return
        self.data, charge = parse_phasespace(code, filename)
        if charge is None:
            charge = 0.0

        self.fname_label.setText(filename)

    def set_control_panel(self):
        """"""
        self.control_panel.setFixedWidth(150)

        phasespace_options = ["x", "xp", "y", "yp", "t", "p"]

        x_label = QLabel('x-axis', self)
        self.xlist.setFixedSize(60, 150)
        self.xlist.addItems(phasespace_options)
        self.xlist.itemClicked.connect(self.update_plot)

        y_label = QLabel('y-axis', self)
        self.ylist.setFixedSize(60, 150)
        self.ylist.addItems(phasespace_options)
        self.ylist.itemClicked.connect(self.update_plot)

        self.cb_cloudplot = QCheckBox('Cloud plot', self)

        gp1 = QGroupBox()
        self.rb_markersize = QRadioButton('Marker size')
        self.sl_markersize = QSlider(Qt.Horizontal)
        self.sl_markersize.setTickPosition(QSlider.TicksBothSides)
        self.sl_markersize.setMinimum(1)
        self.sl_markersize.setMaximum(15)
        self.sl_markersize.setTickInterval(1)
        self.sl_markersize.setSingleStep(1)

        vbox = QVBoxLayout()
        vbox.addWidget(self.rb_markersize)
        vbox.addWidget(self.sl_markersize)
        gp1.setLayout(vbox)

        layout = QGridLayout()
        layout.addWidget(x_label, 0, 0, 1, 1)
        layout.addWidget(self.xlist, 1, 0, 1, 1)
        layout.addWidget(y_label, 0, 1, 1, 1)
        layout.addWidget(self.ylist, 1, 1, 1, 1)
        layout.addWidget(self.cb_cloudplot, 2, 0, 1, 1)
        layout.addWidget(gp1, 3, 0, 1, 2)
        layout.setRowStretch(4, 1)
        self.control_panel.setLayout(layout)

    def update_plot(self):
        """Update the phase-space plot."""
        if not self.xlist.currentItem() or not self.ylist.currentItem():
            return

        if self.data is not None:
            var_x = self.xlist.currentItem().text()
            var_y = self.ylist.currentItem().text()
            x = get_column_by_name(self.data, var_x)
            y = get_column_by_name(self.data, var_y)

            self.psplot.plot(x, y, pen=None, symbol='o', clear=True)

            self.psplot.setLabel('left', var_x)
            self.psplot.setLabel('bottom', var_y)

    def _add_actions(self, target, actions):
        """"""
        for action in actions:
            if action is None:
                target.addSeparator()
            else:
                target.addAction(action)
