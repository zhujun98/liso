# LISO

[![Lates Release](https://img.shields.io/github/v/release/zhujun98/liso)](https://github.com/zhujun98/liso/releases)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Build Status](https://dev.azure.com/zhujun981661/zhujun981661/_apis/build/status/zhujun98.liso?branchName=master)](https://dev.azure.com/zhujun981661/zhujun981661/_build/latest?definitionId=1&branchName=master)
[![Python package](https://github.com/zhujun98/liso/actions/workflows/python-package.yml/badge.svg)](https://github.com/zhujun98/liso/actions/workflows/python-package.yml)
[![codecov.io](https://codecov.io/github/zhujun98/liso/coverage.svg?branch=master)](https://codecov.io/github/zhujun98/liso?branch=master)
[![Documentation](https://img.shields.io/readthedocs/liso)](https://liso.readthedocs.io/en/latest/)

Author: Jun Zhu, zhujun981661@gmail.com

LISO (**LI**nac **S**imulation and **O**ptimization) is a library which provides
a unified interface for numerical simulations, experiments and data management
in the era of big data and artificial intelligence. LISO was initially created only
for simulation and optimization back in 2017. After a more than two years interruption, 
it is live again but aims at:

1. providing a high-level API to run a large numbers of beam dynamics simulations using a combination
of different codes;
2. providing a unified IO for the simulated and experimental data;
3. providing an interface for deep learning and deep reinforcement learning studies on accelerator physics.


## Documentation

The full documentation can be found at 

https://liso.readthedocs.io/

## Getting started

### Installation

```sh
$ pip install liso
```

### Use LISO in your experiments

#### Acquiring data

```py
from liso import EuXFELInterface, MachineScan
from liso import doocs_channels as dc


m = EuXFELInterface()

m.add_control_channel(dc.FLOAT, 'XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/PHASE.SAMPLE')
m.add_control_channel(dc.FLOAT, 'XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/AMPL.SAMPLE')
m.add_control_channel(dc.FLOAT, 'XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/PHASE.SAMPLE')
m.add_control_channel(dc.FLOAT, 'XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/AMPL.SAMPLE')

m.add_diagnostic_channel(dc.IMAGE, 'XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ',
                         shape=(1750, 2330), dtype='uint16')

sc = MachineScan(m)

sc.scan(4000, folder='my_exp', n_tasks=8)
```

#### Reading experimental data from files

```py
from liso import open_run

run = open_run('my_exp/r0001')

# Get an overview of the run
run.info()

# Loop over a run
for pid, data in run:
    pass

# Access a single macropulse
# - by id
pid, data = run.from_id(123456789)
# - by index
pid, data = run.from_index(0)

# Access a channel data
ch_data = run.channel('XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/PHASE.SAMPLE')
# and convert it to a numpy array
ch_data_array = ch_data.numpy()
```

### Use LISO to run simulations

#### Building a linac

```py
linac = Linac(2000)

linac.add_beamline('astra',
                   name='injector',
                   swd='../astra_files',
                   fin='injector.in',
                   template='injector.in.000',
                   pout='injector.0450.001')
```

#### Running a parameter scan or a jitter study

```py
from liso import LinacScan


sc = LinacScan(linac)

sc.add_param('gun_gradient', start=120, stop=140, num=10, sigma=-0.001)
sc.add_param('gun_phase', start=-10, stop=10, num=20, sigma=0.1)

sc.scan(folder="my_scan_data")
```

#### Reading simulated data from files

```py
from liso import open_sim

sim = open_sim('my_scan_data')

# Get an overview of the simulation
sim.info()

# Loop over the simulated data
for sid, data in sim:
    pass

# Access a single simulation
# - by id
sid, data = sim.from_id(1)
# - by index
sid, data = sim.from_index(0)

# Access a phasespace data
ch_data = sim.channel('injector/out', ['x', 'px'])
# and convert it to a numpy array
ch_data_array = ch_data.numpy()
```

### Cite LISO

A BibTeX entry that you can use to cite it in a publication:

    @misc{liso,
      Author = {J. Zhu},
      Title = {LISO},
      Year = {2020},
      publisher = {GitHub},
      journal = {https://github.com/zhujun98/liso},
      version = {...}
    }
