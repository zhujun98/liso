# LISO

Jun Zhu

---

## Introduction

'LISO', **Li**nac **S**imulation and **O**ptimization, is an API for various beam dynamics and FEL codes like ASTRA, IMPACT-T, IMPACT-Z and GENESIS.

## Dependencies

Python3 >= 3.5

#### Beam dynamis and FEL codes:
- [ASTRA](http://www.desy.de/~mpyflo/)
- [IMPACT-T](http://portal.nersc.gov/project/m669/IMPACT-T/)
- [IMPACT-Z]
- [GENESIS]

## Installation

### LISO

```sh
$ git clone https://github.com/zhujun98/liso.git
$ cd liso
$ python setup.py install
```
It is recommended to uninstall the old version before installing a new one. To uninstall:

```sh
$ python setup.py install --record files.txt
$ cat files.txt | xargs rm -rf
```

### Optional 3rd party optimization libraries

#### pyOpt
```sh
$ git clone https://github.com/zhujun98/pyOpt
$ cd pyOpt
$ python setup.py install
```

## GUI

The GUI is based on [PyQt5](https://www.riverbankcomputing.com/software/pyqt/download5) and [pyqtgraph](http://www.pyqtgraph.org/).

```py
from liso import gui

gui()
```

![alt text](misc/GUI_v1.png)


## Optimization

LISO has its own single- and multi-objective optimizers. It also provides interfaces for 3rd party optimizers.

### Optimizers

#### Local unconstrained optimizers
##### Nelder-Mead

#### Local constrained optimizers
##### SDPEN (from pyOpt)

#### Single-objective global optimizers
##### ALPSO

#### Multi-objective optimizers
##### MOPSO




