Optimizers
==========

LISO has its own single- and multi-objective optimizers. It also provides interfaces for 3rd party optimizers.

.. currentmodule:: liso.optimizers.optimizer

.. autoclass:: Optimizer
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: __call__

Local unconstrained optimizers
------------------------------


Nelder-Mead
~~~~~~~~~~~

.. currentmodule:: liso.optimizers
.. autoclass:: NelderMead
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: __call__

Local constrained optimizers
----------------------------

SDPEN (from pyOpt)
~~~~~~~~~~~~~~~~~~

see `SDPEN <http://www.pyopt.org/reference/optimizers.sdpen.html>`_

Single-objective global optimizers
----------------------------------

ALPSO
~~~~~

.. autoclass:: ALPSO
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: __call__

Multi-objective optimizers
--------------------------

MOPSO
~~~~~