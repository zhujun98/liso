Scanning Machine Parameters
===========================

.. _parameter scan with doocs:

Parameter scan with DOOCS
-------------------------

In this section, we will be performing parameter scan for the Injector of the
European XFEL.

.. code-block:: py

    from liso import EuXFELInterface, MachineScan
    from liso import doocs_channels as dc


    m.add_control_channel('XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/PHASE.SAMPLE', dc.FLOAT,
                          write_address='XFEL.RF/LLRF.CONTROLLER/CTRL.GUN.I1/SP.PHASE')
    m.add_control_channel('XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/PHASE.SAMPLE', dc.FLOAT,
                          write_address='XFEL.RF/LLRF.CONTROLLER/CTRL.A1.I1/SP.PHASE')
    m.add_control_channel('XFEL.RF/LLRF.CONTROLLER/VS.AH1.I1/PHASE.SAMPLE', dc.FLOAT,
                          write_address='XFEL.RF/LLRF.CONTROLLER/CTRL.AH1.I1/SP.PHASE')

    m.add_diagnostic_channel(dc.FLOAT, 'XFEL.SDIAG/BAM/47.I1/LOW_CHARGE_SINGLEBUNCH_ARRIVAL_TIME.1')

    sc = MachineScan(m)

    sc.add_param('XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/PHASE.SAMPLE', lb=-3, ub=3)
    sc.add_param('XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/PHASE.SAMPLE', lb=-3, ub=3)
    sc.add_param('XFEL.RF/LLRF.CONTROLLER/VS.AH1.I1/PHASE.SAMPLE', lb=182, ub=186)

    sc.scan(1000, n_tasks=8)

As shown in the above example, besides defining a :ref:`doocs interface`,
one must provide the `write_address` for a control channel in order to
successfully perform a parameter scan. To understand the difference between
the write address of a control channel and the address (we will call it read
address in the following to clearly distinguish it from the write address) of
a control channel, please pay attention to the following example:

.. code-block:: bash

    # write address of the RF phase of the gun
    >>> pydoocs.read("XFEL.RF/LLRF.CONTROLLER/CTRL.GUN.I1/SP.PHASE")
    {'data': -43.0, 'type': 'FLOAT', 'timestamp': 1632771370.628085, 'macropulse': 1180101532, 'miscellaneous': {}}
    >>> pydoocs.read("XFEL.RF/LLRF.CONTROLLER/CTRL.GUN.I1/SP.PHASE")
    {'data': -43.0, 'type': 'FLOAT', 'timestamp': 1632771370.628085, 'macropulse': 1180101532, 'miscellaneous': {}}

    # read address of the RF phase of the gun
    >>> pydoocs.read("XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/PHASE.SAMPLE")
    {'data': -42.88679885864258, 'type': 'FLOAT', 'timestamp': 1632774228.738694, 'macropulse': 1180130114, 'miscellaneous': {}}
    >>> pydoocs.read("XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/PHASE.SAMPLE")
    {'data': -42.90991973876953, 'type': 'FLOAT', 'timestamp': 1632774231.039035, 'macropulse': 1180130137, 'miscellaneous': {}}


You should have noticed that, unlike the result obtained from the read address,
the 'data', 'timestamp' and 'macropulse' of the result obtained from the write
address never change. In a word, the 'data' of the write address was set only
once sometime in the past, while the 'data' of the read address reflects the
change and is sampled from the realtime measurement.

After defining the interface, we need to define the channels to be scanned and
how should they be scannned. For example,

.. code-block:: py

    sc.add_param('XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/PHASE.SAMPLE', lb=-3, ub=3)

defines that the RF phase of the gun will be randomly sampled between -3 and 3.
For different types of scanning parameters please check :ref:`scan parameters api`.
It should be noted that the read address of a control channel should be passed
as the first argument to the method :py:meth:`~liso.scan.machine_scan.MachineScan.add_param`
although the new values will be written to the write address of it.

By default, the scan output is stored in a "run" folder in the current
directory. The number of "run" folder is generated in sequence starting from
"r0001". For how to read out the result, please refer to :ref:`reading experimental data`.

For more details, check the `example <https://github.com/zhujun98/liso/tree/master/examples/xfel_experiment>`_.
