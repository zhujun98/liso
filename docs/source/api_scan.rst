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


.. _scan parameters:

Scan parameters
---------------

.. autofunction:: liso.scan.linac_scan.LinacScan.add_param
    :noindex:

There are three different kinds of `ScanParam` types:

.. currentmodule:: liso.scan.scan_param

.. autoclass:: StepParam

    .. automethod:: __init__

.. autoclass:: SampleParam

    .. automethod:: __init__

.. autoclass:: JitterParam

    .. automethod:: __init__
