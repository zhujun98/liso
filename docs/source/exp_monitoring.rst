Monitoring
==========

.. _monitoring with doocs:

Monitoring with DOOCS
---------------------

LISO provides the command line tool `liso-doocs-monitor` for quickly checking
data from DOOCS channels:

.. code-block:: bash

    usage: liso-doocs-monitor [-h] [--file FILE] [--correlate] [channels]

    positional arguments:
      channels              DOOCS channel addresses separated by comma.

    optional arguments:
      -h, --help            show this help message and exit
      --file FILE, -f FILE  Read DOOCS channel addresses from the given file.
      --correlate           Correlate all channel data by macropulse ID.


For example, simply type the following in a terminal to monitor two channels:

.. code-block:: bash

    liso-doocs-monitor XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/PHASE.SAMPLE,XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/PHASE.SAMPLE

If you have many channels to monitor, you can read the channel addresses from a file.
An example file `channels.txt` is shown below:

.. code-block:: bash

    XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/PHASE.SAMPLE
    XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/PHASE.SAMPLE
    XFEL.RF/LLRF.CONTROLLER/VS.AH1.I1/PHASE.SAMPLE
    XFEL.DIAG/CHARGE.ML/TORA.25.I1/CHARGE.ALL
    ---
    XFEL.MAGNETS/MAGNET.ML/QI.63.I1D/KICK_MRAD.SP
    XFEL.MAGNETS/MAGNET.ML/QI.64.I1D/KICK_MRAD.SP

Each line is treated as a single address. In order to correlate non-event based
channel data (from slow collectors), you can separate the non-event based channels
from the event based channels by having a line starting with three dashes "---".
Typing the following in a terminal:

.. code-block:: bash

    liso-doocs-monitor -f channels.txt --correlate

It should print out the correlated channel data continuously, for example:

.. code-block:: bash

    --------------------------------------------------------------------------------
    Macropulse ID: 1171436850

    - XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/PHASE.SAMPLE: {'data': -42.78379440307617, 'type': 'FLOAT', 'timestamp': 1631904894.110795, 'macropulse': 1171436850, 'miscellaneous': {}}
    - XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/PHASE.SAMPLE: {'data': 11.663818359375, 'type': 'FLOAT', 'timestamp': 1631904894.11092, 'macropulse': 1171436850, 'miscellaneous': {}}
    - XFEL.RF/LLRF.CONTROLLER/VS.AH1.I1/PHASE.SAMPLE: {'data': 167.68936157226562, 'type': 'FLOAT', 'timestamp': 1631904894.11092, 'macropulse': 1171436850, 'miscellaneous': {}}
    - XFEL.DIAG/CHARGE.ML/TORA.25.I1/CHARGE.ALL: {'data': 0.2488500028848648, 'type': 'FLOAT', 'timestamp': 1631904894.049937, 'macropulse': 1171436850, 'miscellaneous': {}}
    - XFEL.MAGNETS/MAGNET.ML/QI.63.I1D/KICK_MRAD.SP: {'data': 1065.099972442661, 'type': 'DOUBLE', 'timestamp': 1631904894.003501, 'macropulse': 0, 'miscellaneous': {}}
    - XFEL.MAGNETS/MAGNET.ML/QI.64.I1D/KICK_MRAD.SP: {'data': -1.2480009219116742e-07, 'type': 'DOUBLE', 'timestamp': 1631904894.003501, 'macropulse': 0, 'miscellaneous': {}}
    --------------------------------------------------------------------------------

or terminate if it cannot correlate all the channels within a given time.