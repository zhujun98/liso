#!/usr/bin/python
"""
Author: Jun Zhu
"""
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QComboBox
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QAction
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtWidgets import qApp
from PyQt5.QtGui import QIcon

from .phasespace_plot import PhaseSpacePlot, create_phasespace_plot

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600


class PhaseSpacePlotGUI(QMainWindow):
    """"""
    def __init__(self, parent=None):
        """Initialization."""
        super().__init__(parent=parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setWindowTitle('Phase Space Plot')

        self._file_menu = None
        self.create_menu()

        self._fname_label = None
        self.create_fname_label()

        self.x_options = None
        self.y_options = None
        self.options = ["", "x", "xp", "y", "yp", "t", "p"]
        self._selection_panel = None
        self.create_selection_panel(self.options)
        self._main_frame = None
        self._fig = None
        self._ax = None
        self.canvas = None
        self._navi_toolbar = None
        self.create_main_frame()

        self._phasespace_plot = None

        self.show()

    def create_menu(self):
        """Create menu bar."""
        self._file_menu = self.menuBar().addMenu("&File")

        load_file_action = self.create_action("&Open File",
                                              shortcut="Ctrl+O",
                                              slot=self.open_file,
                                              tip="Open particle file")
        quit_action = self.create_action("&Quit", slot=self.close,
                                         shortcut="Ctrl+Q",
                                         tip="Close the application")

        self.add_actions(self._file_menu, (load_file_action, None, quit_action))

    def create_main_frame(self):
        """"""
        self._main_frame = QWidget()

        self._fig = Figure(figsize=(8, 6), dpi=300, tight_layout=True)
        self._ax = self._fig.add_subplot(111)

        self.canvas = FigureCanvas(self._fig)
        self.canvas.setParent(self._main_frame)

        self._navi_toolbar = NavigationToolbar(self.canvas, self._main_frame)

        vbox1 = QVBoxLayout()
        vbox1.addWidget(self._navi_toolbar)
        vbox1.addWidget(self._fname_label)
        vbox1.addWidget(self.canvas)

        vbox2 = QVBoxLayout()
        vbox2.addWidget(self._selection_panel)

        hbox = QHBoxLayout()
        hbox.addLayout(vbox1)
        hbox.addLayout(vbox2)

        self._main_frame.setLayout(hbox)
        self.setCentralWidget(self._main_frame)

    def open_file(self):
        """Initialize a PhaseSpacePlot object."""
        # filename, _ = QFileDialog.getOpenFileName(self, 'Open file', '/home')
        filename = "/home/jun/Projects/LISO/examples/astra_basic/injector.0400.001"
        self._fname_label.setText(filename)
        self._phasespace_plot = create_phasespace_plot(
            'astra', filename, fig=self._fig, ax=self._ax)

    def create_selection_panel(self, options):
        """"""
        self._selection_panel = QWidget()

        x_label = QLabel('x-axis', self)
        self.x_options = QComboBox(self)
        self.x_options.addItems(options)
        self.x_options.currentIndexChanged.connect(self.update_plot)

        y_label = QLabel('y-axis', self)
        self.y_options = QComboBox(self)
        self.y_options.addItems(options)
        self.y_options.currentIndexChanged.connect(self.update_plot)

        layout = QVBoxLayout()

        layout.addWidget(x_label)
        layout.addWidget(self.x_options)
        layout.addWidget(y_label)
        layout.addWidget(self.y_options)
        layout.addStretch()

        self._selection_panel.setLayout(layout)

    def create_fname_label(self):
        """Add label for showing the file currently being processed."""
        self._fname_label = QLabel("", self)
        self._fname_label.setFixedWidth(WINDOW_WIDTH - 200)
        self._fname_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self._fname_label.move(150, 40)

    def update_plot(self):
        """"""
        x = self.x_options.currentText()
        y = self.y_options.currentText()
        if x and y and isinstance(self._phasespace_plot, PhaseSpacePlot):
            self._phasespace_plot.scatter(x, y)
        self.canvas.draw()

    def create_action(self, text,
                      slot=None,
                      shortcut=None,
                      icon=None,
                      tip=None,
                      checkable=False,
                      signal="triggered()"):
        """"""
        action = QAction(text, self)
        if icon is not None:
            action.setIcon(QIcon(":/%s.png" % icon))
        if shortcut is not None:
            action.setShortcut(shortcut)
        if tip is not None:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        if slot is not None:
            action.triggered.connect(slot)
        if checkable:
            action.setCheckable(True)

        return action

    def add_actions(self, target, actions):
        for action in actions:
            if action is None:
                target.addSeparator()
            else:
                target.addAction(action)

        # convert inch to pixel
        # if self.phasespace_plot is not None:
        #     w = self.phasespace_plot.figure.get_size_inches()[0] * self.canvas.figure.dpi
        #     h = self.phasespace_plot.figure.get_size_inches()[1] * self.canvas.figure.dpi
        #     self.phasespace_plot.move(WINDOW_WIDTH - w - 15, WINDOW_HEIGHT - h - 40)