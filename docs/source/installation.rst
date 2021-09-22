Installation
============

Using PyPI
----------

.. code-block:: bash

    pip install liso


From Source
-----------

.. code-block:: bash

    git clone https://github.com/zhujun98/liso.git
    cd liso
    pip install -e .


Install optional 3rd party optimization libraries
-------------------------------------------------

pyOpt
~~~~~

.. code-block:: bash

    $ git clone https://github.com/zhujun98/pyOpt
    $ cd pyOpt
    $ sudo pip3 install .


.. _configuration:

Configuration
-------------

When starting LISO for the first time, a default config file
`$HOME/.liso/config.ini` will be generated. You may need to modify the
executables for different simulation codes.

.. code-block::

    [DEFAULT]
    log_file = liso.log
    opt_log_file = liso_opt.log

    [EXECUTABLE]
    astra = astra
    impactt = ImpactTv1.7linux
    elegant = elegant

    [EXECUTABLE_PARA]
    astra = astra_r62_Linux_x86_64_OpenMPI_1.6.1
    impactt = ImpactTv1.7linuxPara
    elegant =
