Scanning Machine Parameters
===========================

.. _parameter scan with doocs:

Parameter scan with DOOCS
-------------------------

In this section, we will be performing parameter scan for the Injector of the
European XFEL. For how to build a machine interface which uses the DOOCS
control system, please refer to :ref:`doocs interface`.

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


Check how to define different :ref:`scan parameters api`.

By default, the scan output is stored in a "run" folder in the current
directory. The number of "run" folder is generated in sequence starting from
"r0001". For how to read out the result, please refer to :ref:`reading experimental data`.

For more details, check the `example <https://github.com/zhujun98/liso/tree/master/examples/xfel_experiment>`_.
