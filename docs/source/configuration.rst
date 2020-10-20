Configuration
=============

When starting LISO for the first time, a default config file `$HOME/.liso/config.ini` will be
generated. You may need to modify the executables for different simulation codes
under sections `[EXECUTABLE]` and `[EXECUTABLE_PARA]`.

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