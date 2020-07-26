"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import sys

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QHBoxLayout

from .plot_guis import PhaseSpacePlotGUI, LinePlotGUI
from .optimizer_gui import OptimizerGUI


WINDOW_HEIGHT = 80
WINDOW_WIDTH = 800
BUTTON_HEIGHT = 60


def get_style_sheet(bg_color):
    """Return the string for setStyleSheet().

    :param bg_color: string
        Color Hex Color Codes.
    """
    style_sheet = 'QPushButton {color: white; font: bold; padding: 5px; ' \
                  + 'background-color: ' + bg_color + '}'
    return style_sheet


class MainGUIButton(QPushButton):
    """Inherited from QPushButton."""
    def __init__(self, name, parent=None):
        """Initialization."""
        super().__init__(name, parent)
        self.setMinimumHeight(BUTTON_HEIGHT)


class MainGUI(QWidget):
    """The main GUI.

    It contains buttons for opening child GUIs with different
    functionality.
    """
    def __init__(self, screen_size=None):
        """Initialization."""
        super().__init__()

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setWindowTitle('LISO Visualization GUI')

        self.create_button_layout()

        if screen_size is None:
            self.move(0, 0)
        else:
            self.move(screen_size.width()/2 - WINDOW_WIDTH/2,
                      screen_size.height()/20)

    def create_button_layout(self):
        """Create buttons."""
        # BUtton for linac layout plot.
        layout_btn = MainGUIButton('Linac Layout', self)
        layout_btn.setStyleSheet(get_style_sheet("#0288CF"))

        # Button for phasespace plot.
        ps_btn = MainGUIButton('Phasespace\n Plot', self)
        ps_btn.setStyleSheet(get_style_sheet("#832E01"))
        ps_btn.clicked.connect(self.open_new_phasespace_plot)

        # Button for line plot.
        line_btn = MainGUIButton('Line Plot', self)
        line_btn.setStyleSheet(get_style_sheet("#CA5504"))
        line_btn.clicked.connect(self.open_new_line_plot)

        # Button for optimization visualization.
        optmization_btn = MainGUIButton('Optimization\n Tracker', self)
        optmization_btn.setStyleSheet(get_style_sheet("#7DB039"))

        # Button for jitter study visualization
        jitter_btn = MainGUIButton('Jitter Tracker', self)
        jitter_btn.setStyleSheet(get_style_sheet("#222760"))

        # Button for visualizing optimizer testing
        optimizer_btn = MainGUIButton('Optimizer\n Visualization', self)
        optimizer_btn.setStyleSheet(get_style_sheet("#C4A464"))
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
        PhaseSpacePlotGUI().show()  # Do not assign parent which causes memory leak

    def open_new_line_plot(self):
        """Open a new window for line plot."""
        LinePlotGUI().show()

    def open_new_optimizer_visualizer(self):
        """Open a new window for line plot."""
        OptimizerGUI().show()


def main_gui():
    app = QApplication(sys.argv)
    screen_size = app.primaryScreen().size()
    ex = MainGUI(screen_size=screen_size)
    app.exec_()
    # print('\n'.join(repr(w) for w in app.allWidgets()))

