Parameter Scan
==============


Run a scan
----------

.. code-block:: py

    from liso import LinacScan


    sc = LinacScan(linac)

    sc.add_param('gun_gradient', start=120, stop=140, num=10, sigma=-0.001)
    sc.add_param('gun_phase', start=-10, stop=10, num=20, sigma=0.1)
    sc.add_param('tws_gradient', lb=25, ub=35)
    sc.add_param('tws_phase', value=-90, sigma=0.1)

    sc.scan(folder="my_scan_data")
