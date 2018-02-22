# LinacOpt

Jun Zhu

---

## Introduction

An object-oriented Python API for beam dynamics optimization with ASTRA and IMPACT-T.

<img src="./miscs/problem_definition.png" width="480"/>

## Dependencies

- Python3 >= 3.5
- [pyOpt](http://www.pyopt.org/) >= 1.2.0
- [ASTRA](http://www.desy.de/~mpyflo/)
- [IMPACT-T](http://portal.nersc.gov/project/m669/IMPACT-T/)

## Installation

- Download and install pyOpt
```sh
$ git clone https://github.com/zhujun98/pyOpt
$ cd pyOpt
$ python setup.py install
```

- Download and install LinacOpt
```sh
$ git clone http://github.com/zhujun98/linacopt.git
$ cd linacopt
$ python setup.py install
```
For parallel version
```sh
$ sudo pip3 install mpi4py
```


## Optimizers

### Global optimizers: 

#### [Augmented Lagrangian Particle Swarm Optimizer](http://www.pyopt.org/reference/optimizers.alpso.html#module-pyALPSO)



#### [Non Sorting Genetic Algorithm II](http://www.pyopt.org/reference/optimizers.nsga2.html#module-pyNSGA2)

### Local search optimizers:

#### [SDPEN](http://www.pyopt.org/reference/optimizers.sdpen.html#module-pySDPEN)


## Common problems and tips

- Do not use very deep directory to run the simulation. Otherwise the name of the output file may be truncated! (This seems to be a problem with FORTRAN).

- Be careful about the number of grids (e.g. nrad and nlong_in in ASTRA)!!! For example, when you are optimizing the emittance of a gun, the optimizer may go over the working point with a very small laser spot size. If the number of grids is too small, it may underestimate the space-charge effects. However, the thermal emittance decreases as the laser spot size decreases. Therefore, if you do not have enough grids, you may get the wrong result in this case. My experience is that the longitudinal grid number is more important.

- The parallel version of ASTRA will be stuck at some crazy working points where a lot of particles are lost. I set a 'time_out' parameter which will kill the simulation after a certain time (the optimization will continue). The default value of 'time_out' is 1200 s.


## Uninstall

```sh
$ python setup.py install --record files.txt
$ cat files.txt | xargs rm -rf
```


