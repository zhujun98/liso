Optimization
============

Guid for beginners
------------------

- Test your optimization problem without collective effects first.
- Try to find a trade-off between speed and accuracy. In ASTRA simulation, for instance, the knobs are
"H_MAX", "MAX_SCALE", "MAX_COUNT", "NRAD", "NLONG_IN", etc.
- Run some small optimization problems first to familiarize yourself with different optimizers and your
problem.
- Your "Objective" should be much larger than "Absolute tolerance" in the optimizer setup.

.. currentmodule:: liso.operation
.. autoclass:: Operation
    :show-inheritance:


.. currentmodule:: liso.optimization.linac_optimization

.. autoclass:: Optimization
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: add_var
    .. automethod:: del_var
    .. automethod:: add_covar
    .. automethod:: del_covar
    .. automethod:: add_obj
    .. automethod:: del_obj
    .. automethod:: add_econ
    .. automethod:: del_econ
    .. automethod:: add_icon
    .. automethod:: del_icon
    .. automethod:: solve

.. autoclass:: LinacOptimization
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: solve