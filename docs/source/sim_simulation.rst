Simulating a Linac
==================

In this section, we will be building various beamlines for simulation using
one or more beam dynamics codes and running the simulation. Although running
a simulation with a single beam dynamics code can be done easily without
Python, this is required to run further parameter scan and optimization in LISO.

.. _injector simulation ASTRA:

Building an injector using ASTRA
--------------------------------

Prerequisite: `ASTRA <https://www.desy.de/~mpyflo/>`_.

.. code-block:: python

    from liso import Linac


    # Instantiate a Linac object
    linac = Linac()

    # Add a beamline which is simulated by ASTRA
    linac.add_beamline('astra',
                       name='gun',
                       swd='../astra_files',
                       fin='injector.in',
                       template='injector.in.000',
                       pout='injector.0450.001')

    # Define parameters which are defined within a pair of angle bracket in the
    # template file.
    params = {
        'laser_spot': 0.1,
        'main_sole_b': 0.2,
    }

    # Run the simulation (test whether everything is set up properly)
    linac.run(params)

For more details, check the `examples <https://github.com/zhujun98/liso/tree/master/examples/astra_basic>`_.

Read more about :ref:`injector optimization ASTRA`.

Read more about :ref:`injector scan ASTRA`.