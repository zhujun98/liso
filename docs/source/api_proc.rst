Data processing
===============

.. currentmodule:: liso.proc

Data parsing
++++++++++++

.. autofunction:: parse_astra_phasespace

.. autofunction:: parse_astra_line

.. autofunction:: parse_impactt_phasespace

.. autofunction:: parse_impactt_line


Phasespace analysis
+++++++++++++++++++

.. autoclass:: Phasespace
    :show-inheritance:

    .. automethod:: __init__
    .. automethod:: slice
    .. automethod:: cut_halo
    .. automethod:: cut_tail
    .. automethod:: rotate
    .. automethod:: analyze

.. autofunction:: compute_canonical_emit

.. autofunction:: compute_current_profile

.. autofunction:: compute_twiss
