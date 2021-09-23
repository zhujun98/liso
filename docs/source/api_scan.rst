Scan
====

.. autoclass:: liso.scan.linac_scan.LinacScan
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: add_param
    .. automethod:: scan


.. autoclass:: liso.scan.machine_scan.MachineScan
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: add_param
    .. automethod:: scan


.. _scan parameters api:

Scan parameters
---------------

There are three different kinds of `ScanParam` types:

.. currentmodule:: liso.scan.scan_param

.. autoclass:: ScanParam

.. autoclass:: StepParam
    :show-inheritance:

    .. automethod:: __init__

.. autoclass:: SampleParam
    :show-inheritance:

    .. automethod:: __init__

.. autoclass:: JitterParam
    :show-inheritance:

    .. automethod:: __init__
