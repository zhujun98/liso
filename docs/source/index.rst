Welcome to LISO's documentation!
================================


Introduction
------------

LISO (**LI**\ nac **S**\ imulation and **O**\ ptimization) is a library which provides
a unified interface for numerical simulations, experiments and data management
in the era of big data and artificial intelligence. LISO was initially created only
for simulation and optimization back in 2017. After a more than two years interruption,
it is live again but aims at:

- providing a high-level API to run a large numbers of beam dynamics simulations using a combination
  of different codes;
- providing a unified IO for the simulated and experimental data;
- providing an interface for deep learning and deep reinforcement learning studies on accelerator physics.


.. toctree::
   :maxdepth: 1
   :caption: INSTALLATION:

   installation

.. toctree::
   :maxdepth: 1
   :caption: SIMULATION:

   sim_simulation
   sim_optimization
   sim_scan
   sim_reading_data

.. toctree::
   :maxdepth: 1
   :caption: EXPERIMENT:

   exp_data_recording
   exp_scan
   exp_reading_data

.. toctree::
   :maxdepth: 1
   :caption: API REFERENCE:

   api_experiment
   api_io
   api_optimization
   api_optimizers
   api_proc
   api_scan
   api_simulation
   api_visualization

.. toctree::
   :maxdepth: 1
   :caption: TUTORIALS:

   notebooks/sim_reading_data_in_hdf5
   notebooks/exp_reading_data_in_hdf5