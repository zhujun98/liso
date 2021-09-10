Scanning Machine Parameters
===========================

.. _injector scan EuXFEL:

Performing parameter scan for the injector of EuXFEL
----------------------------------------------------

.. code-block:: py

    from liso import EuXFELInterface, MachineScan
    from liso import doocs_channels as dc

    m = EuXFELInterface()

    m.add_control_channel(dc.FLOAT, 'XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/PHASE.SAMPLE')
    m.add_control_channel(dc.FLOAT, 'XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/AMPL.SAMPLE')
    m.add_control_channel(dc.FLOAT, 'XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/PHASE.SAMPLE')
    m.add_control_channel(dc.FLOAT, 'XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/AMPL.SAMPLE')

    # non-event based data
    m.add_control_channel(dc.FLOAT, 'XFEL.MAGNETS/MAGNET.ML/QI.63.I1D/KICK_MRAD.SP')
    m.add_control_channel(dc.FLOAT, 'XFEL.MAGNETS/MAGNET.ML/QI.64.I1D/KICK_MRAD.SP')

    m.add_diagnostic_channel(dc.FLOAT, 'XFEL.DIAG/CHARGE.ML/TORA.25.I1/CHARGE.ALL')

    m.add_diagnostic_channel(dc.IMAGE, 'XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ',
                             shape=(1750, 2330), dtype='uint16', no_event=True)

    sc = MachineScan(m)

    sc.add_param('XFEL.RF/LLRF.CONTROLLER/CTRL.GUN.I1/SP.PHASE',
                 readout='XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/PHASE.SAMPLE',
                 tol=0.02,
                 lb=-3, ub=3)
    sc.add_param('XFEL.RF/LLRF.CONTROLLER/CTRL.A1.I1/SP.PHASE',
                 readout='XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/PHASE.SAMPLE',
                 tol=0.02,
                 lb=-3, ub=3)

    sc.scan(1000, tasks=8)


Check how to define different :ref:`scan parameters`.

By default, the scan output will be stored in the current directory. For how to
read out the result, please refer to :ref:`reading hdf5 exp`.
