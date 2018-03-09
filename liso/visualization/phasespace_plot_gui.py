#!/usr/bin/python
"""
Author: Jun Zhu
"""
import sys

import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QComboBox
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QAction
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtWidgets import qApp
from PyQt5.QtGui import QIcon


WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600


class DynamicMplCanvas(FigureCanvas):
    """A canvas that updates itself every second with a new plot."""
    def __init__(self, parent=None, width=6.4, height=4.8):
        """"""
        fig = Figure(figsize=(width, height), tight_layout=True)

        super().__init__(fig)
        self.ax = self.figure.add_subplot(111)
        super().setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        super().updateGeometry()

        self.setParent(parent)

    def update_figure(self, phasespace_plot, x, y):
        """"""
        self.ax.cla()
        phasespace_plot.plot(x, y, ax=self.ax)
        self.draw()


class PhaseSpacePlotGUI(QMainWindow):
    """"""
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setWindowTitle('Phase-space Plots')
        self.statusBar().showMessage('Ready')

        self._phasespace_plot = None

        # -------------------------------------------------------------
        # Action buttons
        # -------------------------------------------------------------
        # Open the phasespace file
        open_action = QAction(QIcon.fromTheme('document-open'), 'Open', self)
        open_action.triggered.connect(self.import_data)
        # Save the image to file
        save_action = QAction(QIcon.fromTheme('image-save-as'), 'Save as', self)
        save_action.triggered.connect(self.save_image)
        # Refresh data and plots
        refresh_action = QAction(QIcon.fromTheme('view-refresh'), 'Refresh', self)

        # -------------------------------------------------------------
        # Label for the file currently being processed
        # -------------------------------------------------------------
        self.fname_label = QLabel("", self)
        self.fname_label.setFixedWidth(WINDOW_WIDTH - 200)
        self.fname_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.fname_label.move(150, 40)

        toolbar = self.addToolBar('Open')
        toolbar.addAction(open_action)
        toolbar.addAction(save_action)
        toolbar.addAction(refresh_action)

        # -------------------------------------------------------------
        # Two drop boxes
        # -------------------------------------------------------------
        vars = ["", "x", "xp", "y", "yp", "t", "delta"]
        x0 = 10
        y0 = 60
        y_space = 40
        x_label = QLabel('x-axis', self)
        x_label.move(x0, y0)

        y0 += y_space
        self.x_axis = QComboBox(self)
        self.x_axis.addItems(vars)
        self.x_axis.move(x0, y0)

        y0 += y_space
        y_label = QLabel('y-axis', self)
        y_label.move(x0, y0)

        y0 += y_space
        self.y_axis = QComboBox(self)
        self.y_axis.addItems(vars)
        self.y_axis.move(x0, y0)

        self.x_axis.currentIndexChanged.connect(self._update_plot)
        self.y_axis.currentIndexChanged.connect(self._update_plot)

        # -------------------------------------------------------------
        # The plot
        # -------------------------------------------------------------
        self.canvas = DynamicMplCanvas(self)
        # convert inch to pixel
        w = self.canvas.figure.get_size_inches()[0] * self.canvas.figure.dpi
        h = self.canvas.figure.get_size_inches()[1] * self.canvas.figure.dpi
        self.canvas.move(WINDOW_WIDTH - w - 15,
                         WINDOW_HEIGHT - h - 40)

        self.show()

    def import_data(self):
        """"""
        filename, _ = QFileDialog.getOpenFileName(self, 'Open file', '/home')
        self.fname_label.setText(filename)

    def save_image(self):
        """"""
        pass

    def _update_plot(self):
        """"""
        x = self.x_axis.currentText()
        y = self.y_axis.currentText()
        if not x or not y:
            return

        self.canvas.update_figure(self._phasespace_plot, x, y)
