Parameter Scan
==============


Run a parameter scan
--------------------

An example script of running a parameter scan is given below:

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

By default, the scan output will be stored in the current directory. For how to
read out the result, please refer to `Reading Simulated Scan Data Files <./sim_reading_scan_files.ipynb>`_.
