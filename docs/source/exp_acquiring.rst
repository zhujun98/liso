Acquiring
==========

.. _data acquisition with doocs:

Data acquisition with DOOCS
---------------------------

In this section, we will be performing data acquisition for the injector of
the European XFEL. For how to build a machine interface which uses the DOOCS
control system, please refer to :ref:`interface with doocs`.

Once a machine interface has been built, one can simply acquire the data and
save it to files by:

.. code-block:: py

   m.acquire()


By default, the scan output is stored in a "run" folder in the current
directory. The number of "run" folder is generated in sequence starting from
"r0001". For how to read out the result, please refer to :ref:`reading hdf5 exp`.

For more details, check the `example <https://github.com/zhujun98/liso/tree/master/examples/xfel_experiment>`_.
