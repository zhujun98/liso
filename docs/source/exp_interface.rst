Machine Interface
=================

.. _interface with doocs:

Interface with DOOCS
--------------------

LISO provides the `MachineInterface` abstraction to allow interacting with a
real machine through its control system. `DOOCS <https://doocs-web.desy.de/index.html>`_,
the Distributed Object-Oriented Control System, is used ubiquitously at DESY.

An example code snippet is shown below:

.. code-block:: py

    from liso import EuXFELInterface
    from liso import doocs_channels as dc

    m = EuXFELInterface()

    m.add_control_channel(dc.FLOAT,
                          'XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/PHASE.SAMPLE',
                          'XFEL.RF/LLRF.CONTROLLER/CTRL.GUN.I1/SP.PHASE')

    m.add_control_channel(dc.FLOAT, 'XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/AMPL.SAMPLE')

    m.add_control_channel(dc.FLOAT, 'XFEL.MAGNETS/MAGNET.ML/QI.63.I1D/KICK_MRAD.SP', no_event=True)
    m.add_control_channel(dc.FLOAT, 'XFEL.MAGNETS/MAGNET.ML/QI.64.I1D/KICK_MRAD.SP', non_event=True)

    m.add_diagnostic_channel(dc.FLOAT, 'XFEL.DIAG/CHARGE.ML/TORA.25.I1/CHARGE.ALL')
    m.add_diagnostic_channel(dc.IMAGE, 'XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ',
                             shape=(1750, 2330), dtype='uint16')


Read more about :ref:`monitoring with DOOCS`.

Read more about :ref:`parameter scan with DOOCS`.
