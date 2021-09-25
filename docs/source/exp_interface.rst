Machine Interface
=================

.. _doocs interface:

Interface with DOOCS
~~~~~~~~~~~~~~~~~~~~

LISO provides the `MachineInterface` abstraction to allow interacting with a
real machine through its control system. `DOOCS <https://doocs-web.desy.de/index.html>`_,
the Distributed Object-Oriented Control System, is used ubiquitously at DESY.

An example code snippet is shown below:

.. code-block:: py

    from liso import EuXFELInterface
    from liso import doocs_channels as dc

    m = EuXFELInterface()

    m.add_control_channel('XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/PHASE.SAMPLE', dc.FLOAT)
    m.add_control_channel('XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/AMPL.SAMPLE', dc.FLOAT)

    m.add_control_channel('XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/PHASE.SAMPLE', dc.FLOAT)
    m.add_control_channel('XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/AMPL.SAMPLE', dc.FLOAT)

    m.add_control_channel('XFEL.MAGNETS/MAGNET.ML/QI.63.I1D/KICK_MRAD.SP', dc.DOUBLE, non_event=True)
    m.add_control_channel('XFEL.MAGNETS/MAGNET.ML/QI.64.I1D/KICK_MRAD.SP', dc.DOUBLE, non_event=True)

    m.add_diagnostic_channel('XFEL.DIAG/CHARGE.ML/TORA.25.I1/CHARGE.ALL', dc.FLOAT)
    m.add_diagnostic_channel('XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ', dc.IMAGE,
                             shape=(1750, 2330), dtype='uint16', no_event=True)


Read more about :ref:`monitoring with doocs`.

Read more about :ref:`parameter scan with doocs`.
