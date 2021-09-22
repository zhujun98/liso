Scanning Linac Parameters
=========================

.. _injector scan ASTRA:

Performing parameter scan for an injector using ASTRA
-----------------------------------------------------

In this section, we will be performing parameter scan for an injector
simulated using ASTRA. For how to run a basic simulation with ASTRA, please
refer to :ref:`injector simulation ASTRA`.

.. code-block:: py

    from liso import Linac, LinacScan


    linac = Linac(2000)

    linac.add_beamline('astra',
                       name='gun',
                       swd='../astra_files',
                       fin='injector.in',
                       template='injector.in.000',
                       pout='injector.0450.001')


    sc = LinacScan(linac)

    sc.add_param('gun_gradient', start=120, stop=140, num=10, sigma=-0.001)
    sc.add_param('gun_phase', start=-10, stop=10, num=20, sigma=0.1)
    sc.add_param('tws_gradient', lb=25, ub=35)
    sc.add_param('tws_phase', value=-90, sigma=0.1)

    sc.scan()

Check how to define different :ref:`scan parameters`.

Run the scan by typing:

.. code-block:: bash

    python astra_scan.py


The following information will be printed out in the terminal:

.. code-block:: bash

    INFO -
    ================================================================================
    Linac definition:

    Beamline: gun
    Simulation working directory: /home/jun/Projects/liso/examples/astra_files
    Input file: injector.in
    Input particle file: None
    Output particle file: injector.0450.001
    ================================================================================

    INFO - Starting parameter scan with 4 CPUs. Scan result will be save at /home/jun/Projects/liso/examples/astra_scan/r0001
    INFO -
    ================================================================================
    Scanned parameters:

    Name             Start          Stop          Num          Sigma
    gun/gun_grad   1.2000e+02    1.4000e+02        4        -1.0000e-03
    gun/gun_phas  -1.0000e+01    1.0000e+01        3         1.0000e-01

    Name          Lower bound   Upper bound
    gun/tws_grad   2.5000e+01    3.5000e+01

    Name             Value         Sigma
    gun/tws_phas  -9.0000e+01    1.0000e-01
    ================================================================================

    INFO - Scan 000001: 'gun/gun_gradient' = 120.070288330761, 'gun/gun_phase' = -9.995571223782681, 'gun/tws_gradient' = 31.652915442319234, 'gun/tws_phase' = -90.01709619835886
    INFO - Scan 000002: 'gun/gun_gradient' = 119.86075461457493, 'gun/gun_phase' = -0.08139387742461611, 'gun/tws_gradient' = 33.269910278422664, 'gun/tws_phase' = -90.0154375748651
    INFO - Scan 000003: 'gun/gun_gradient' = 120.02260037875811, 'gun/gun_phase' = 9.89076943291548, 'gun/tws_gradient' = 31.20198773846917, 'gun/tws_phase' = -90.18135938026114
    INFO - Scan 000004: 'gun/gun_gradient' = 126.51000398703732, 'gun/gun_phase' = -10.253028545608556, 'gun/tws_gradient' = 25.3089216280909, 'gun/tws_phase' = -89.96489743607206
    INFO - Scan 000005: 'gun/gun_gradient' = 126.58185626272527, 'gun/gun_phase' = 0.03932701974230741, 'gun/tws_gradient' = 26.345851198480762, 'gun/tws_phase' = -90.1652409860634
    INFO - Scan 000006: 'gun/gun_gradient' = 126.85819172668924, 'gun/gun_phase' = 10.076015458347582, 'gun/tws_gradient' = 26.147351988460457, 'gun/tws_phase' = -89.93689635300981
    INFO - Scan 000007: 'gun/gun_gradient' = 133.47926293544657, 'gun/gun_phase' = -10.074552827274841, 'gun/tws_gradient' = 27.30580361234209, 'gun/tws_phase' = -90.03408839203139
    INFO - Scan 000008: 'gun/gun_gradient' = 133.37476124118928, 'gun/gun_phase' = 0.13249019179700253, 'gun/tws_gradient' = 28.594443157747044, 'gun/tws_phase' = -90.01830630577201
    INFO - Scan 000009: 'gun/gun_gradient' = 133.08019409093532, 'gun/gun_phase' = 10.034770415332911, 'gun/tws_gradient' = 29.014661849286902, 'gun/tws_phase' = -90.00254695475891
    INFO - Scan 000010: 'gun/gun_gradient' = 140.2750893670717, 'gun/gun_phase' = -9.911377437568264, 'gun/tws_gradient' = 34.94958987086205, 'gun/tws_phase' = -90.15719490311994
    INFO - Scan 000011: 'gun/gun_gradient' = 140.11315476410957, 'gun/gun_phase' = -0.15818478995293947, 'gun/tws_gradient' = 32.49524713186926, 'gun/tws_phase' = -90.11206382667498
    INFO - Scan 000012: 'gun/gun_gradient' = 140.01509863571056, 'gun/gun_phase' = 9.972137401245421, 'gun/tws_gradient' = 25.155075425213514, 'gun/tws_phase' = -90.00046445510179
    INFO - Scan finished!


By default, the scan output is stored in a "run" folder in the current
directory. The number of "run" folder is generated in sequence starting from
"r0001". For how to read out the result, please refer to :ref:`reading hdf5 sim`.

For more details, check the `example <https://github.com/zhujun98/liso/tree/master/examples/astra_scan>`_.
