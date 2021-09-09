Optimization
============


Optimizing an injector using ASTRA
----------------------------------

In this section, we will optimize the performance of the injector introduced in:
:ref:`Building an injector using ASTRA`.

.. code-block:: python

    # Instantiate an Optimization object
    opt = LinacOptimization(linac)

    # Add the objective (the horizontal emittance at the end of the 'gun' beamline)
    opt.add_obj('emitx_um', expr='gun.out.emitx', scale=1e6)

    # Add variables with lower boundary (lb) and upper boundary (ub)
    opt.add_var('laser_spot',  value=0.10, lb=0.04, ub=0.3)
    opt.add_var('main_sole_b', value=0.20, lb=0.00, ub=0.4)

    # Instantiate an optimizer
    optimizer = NelderMead()

    # Run the optimization
    opt.solve(optimizer)


For more details, check the `examples <https://github.com/zhujun98/liso/tree/master/examples/astra_basic>`_.