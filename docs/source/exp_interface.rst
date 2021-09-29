Machine Interface
=================

.. _doocs interface:

Interface with DOOCS
~~~~~~~~~~~~~~~~~~~~

LISO provides the `MachineInterface` abstraction to allow interacting with a
real machine through its control system. `DOOCS <https://doocs-web.desy.de/index.html>`_,
the Distributed Object-Oriented Control System, is used ubiquitously at DESY.

An example code snippet is shown below:

.. code-block:: py

    from liso import EuXFELInterface
    from liso import doocs_channels as dc

    m = EuXFELInterface()

    m.add_control_channel('XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/PHASE.SAMPLE', dc.FLOAT)
    m.add_control_channel('XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/AMPL.SAMPLE', dc.FLOAT)

    m.add_control_channel('XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/PHASE.SAMPLE', dc.FLOAT)
    m.add_control_channel('XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/AMPL.SAMPLE', dc.FLOAT)

    m.add_control_channel('XFEL.MAGNETS/MAGNET.ML/QI.63.I1D/KICK_MRAD.SP', dc.DOUBLE, non_event=True)
    m.add_control_channel('XFEL.MAGNETS/MAGNET.ML/QI.64.I1D/KICK_MRAD.SP', dc.DOUBLE, non_event=True)

    m.add_diagnostic_channel('XFEL.DIAG/CHARGE.ML/TORA.25.I1/CHARGE.ALL', dc.FLOAT)
    m.add_diagnostic_channel('XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ', dc.ARRAY,
                             shape=(1750, 2330), dtype='uint16', non_event=True)


Conceptually, DOOCS channels are categorized into control and diagnostic
channels in LISO. A control channel can be set to change the behavior of the
machine, for example, an RF phase and a Quadrupole magnet current. In principle,
the value of a control channel should be a single number. A diagnostic channel
provides the measured result which cannot be changed explicitly, for example,
the image on an OTR screen. In many cases, the value of a diagnostic channel
is an array. If you do not plan to write new values to the machine, for instance,
performing a :ref:`parameter scan with doocs`, you can actually add a
diagnostic channel as a control channel and vise versa. Nevertheless, it is
recommended to follow the above convention since this may affect reading data
from files and further analysis.

Once the interface is defined, we can start to acquire the data defined in the
control and diagnostic channels and save them to files by:

.. code-block:: py

   m.acquire()


By default, the scan output is stored in a "run" folder in the current
directory. The number of "run" folder is generated in sequence starting from
"r0001". For how to read out the result, please refer to :ref:`reading experimental data`.

For more details, check the `example <https://github.com/zhujun98/liso/tree/master/examples/xfel_experiment>`_.


Read more about :ref:`monitoring with doocs`.
