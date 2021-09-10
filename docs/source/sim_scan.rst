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

By default, the scan output will be stored in the current directory. For how to
read out the result, please refer to :ref:`reading hdf5 sim`.

For more details, check the `examples <https://github.com/zhujun98/liso/tree/master/examples/astra_scan>`_.
