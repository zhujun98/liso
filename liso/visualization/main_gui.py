#!/usr/bin/python
"""
Author: Jun Zhu
"""
import sys

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QHBoxLayout

from .phasespace_plot_gui import PhaseSpacePlotGUI
from .line_plot_gui import LinePlotGUI
from .optimizer_gui import OptimizerGUI


class MainGUI(QWidget):
    """The main GUI.

    It contains buttons for opening child GUIs with different
    functionality.
    """
    def __init__(self):
        """Initialization."""
        super().__init__()

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setFixedSize(800, 60)
        self.setWindowTitle('LISO Visualization GUI')
        # self.statusBar().showMessage('Ready')

        self.create_button_layout()

    def create_button_layout(self):
        """Create buttons."""
        # BUtton for linac layout plot.
        layout_btn = QPushButton('Linac\n Layout', self)

        # Button for phasespace plot.
        ps_btn = QPushButton('Phasespace\n Plot', self)
        ps_btn.clicked.connect(self.open_new_phasespace_plot)

        # Button for line plot.
        line_btn = QPushButton('Line\n Plot', self)
        line_btn.clicked.connect(self.open_new_line_plot)

        # Button for optimization visualization.
        optmization_btn = QPushButton('Optimization\n Tracker', self)

        # Button for jitter study visualization
        jitter_btn = QPushButton('Jitter\n Tracker', self)

        # Button for visualizing optimizer testing
        optimizer_btn = QPushButton('Optimizer\n Visualization', self)
        optimizer_btn.clicked.connect(self.open_new_optimizer_visualizer)

        hbox = QHBoxLayout()
        hbox.addWidget(layout_btn)
        hbox.addWidget(ps_btn)
        hbox.addWidget(line_btn)
        hbox.addWidget(optmization_btn)
        hbox.addWidget(jitter_btn)
        hbox.addWidget(optimizer_btn)

        self.setLayout(hbox)

        self.show()

    def open_new_phasespace_plot(self):
        """Open a new window for phase-space plot."""
        PhaseSpacePlotGUI(parent=self).show()

    def open_new_line_plot(self):
        """Open a new window for line plot."""
        LinePlotGUI(parent=self).show()

    def open_new_optimizer_visualizer(self):
        """Open a new window for line plot."""
        OptimizerGUI(parent=self).show()


def main_gui():
    app = QApplication(sys.argv)
    ex = MainGUI()
    sys.exit(app.exec_())
