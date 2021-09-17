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

