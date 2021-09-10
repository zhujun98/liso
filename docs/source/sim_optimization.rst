Optimizing beam parameters
==========================

.. _injector optimization ASTRA:

Optimizing an injector using ASTRA
----------------------------------

In this section, we will be optimizing the performance of the injector
introduced in :ref:`injector simulation ASTRA`.

.. code-block:: python

    from liso import NelderMead

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

The optimizer :class:`liso.optimizers.NelderMead` used above is for
`local optimization. <https://en.wikipedia.org/wiki/Local_search_(optimization)>`_
Besides local optimizers, LISO also provides more powerful optimizers, such
as :class:`liso.optimizers.ALPSO`,
for `global optimization <https://en.wikipedia.org/wiki/Global_optimization>`_.


.. code-block:: python

    from liso import ALPSO

    opt = LinacOptimization(linac)

    opt.add_obj('St', expr='gun.out.St', scale=1e15)

    opt.add_econ('n_pars', expr='gun.out.n', eq=2000)
    opt.add_icon('emitx', expr='gun.out.emitx', scale=1e6,  ub=0.3)
    opt.add_icon('gamma', func=lambda a: a['gun'].out.gamma,  lb=20.0)
    opt.add_icon('max_Sx', func=lambda a: a['gun'].max.Sx*1e3, ub=3.0)

    opt.add_var('laser_spot',  value=0.1, lb=0.04, ub=0.5)
    opt.add_var('main_sole_b', value=0.2, lb=0.00, ub=0.4)
    opt.add_var('gun_phase', value=0.0, lb=-10, ub=10)
    opt.add_var('tws_phase', value=0.0, lb=-90, ub=0)

    optimizer = ALPSO()
    optimizer.swarm_size = 40
    opt.solve(optimizer)


For more details, check the `examples <https://github.com/zhujun98/liso/tree/master/examples/astra_advanced>`_.
