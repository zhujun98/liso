Installation
============

Install LISO
------------

.. code-block:: bash

    $ git clone https://github.com/zhujun98/liso.git
    $ cd liso
    $ sudo pip3 install .
    # To uninstall
    $ sudo pip3 uninstall liso

Set up the environment
----------------------

Modify the configuration file *liso/config.py*!

.. code-block:: python

    ASTRA = "astra"  # Your system command to run ASTRA
    ASTRA_P = "astra_r62_Linux_x86_64_OpenMPI_1.6.1"  # parallel ASTRA
    IMPACTT = "ImpactTv1.7linux"
    IMPACTT_P = "ImpactTv1.7linuxPara"  # paralle IMPACT-T

Install optional 3rd party optimization libraries
-------------------------------------------------

pyOpt
~~~~~

.. code-block:: bash

    $ git clone https://github.com/zhujun98/pyOpt
    $ cd pyOpt
    $ sudo pip3 install .


GUI
~~~

The GUI is based on `PyQt5 <https://www.riverbankcomputing.com/software/pyqt/download5>`_ and `pyqtgraph <http://www.pyqtgraph.org/>`_.
