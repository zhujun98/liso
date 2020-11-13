"""
This is a basic example showing

```
python elegant_basic.py
```

Author: Jun Zhu
"""
from liso import Linac, LinacScan
from liso.logging import logger
logger.setLevel('DEBUG')

linac = Linac(2000)

linac.add_beamline('astra',
                   name='gun',
                   swd='../astra_files',
                   fin='injector.in',
                   template='injector.in.000',
                   pout='injector.0450.001')

linac.add_beamline('elegant',
                   name='tds',
                   swd='../elegant_files',
                   fin='run.ele',
                   template='run.ele.000',
                   pin='input.sdds',
                   pout='lps.out')


sc = LinacScan(linac)

sc.add_param('gun_gradient', 120, 140, num=2, sigma=-0.001)
sc.add_param('gun_phase', -10, 10, num=2, sigma=0.1)
sc.add_param('tws_gradient', 25, 35, num=2)
sc.add_param('tws_phase', -90, -60, num=3)

sc.scan(1, folder="my_scan_data")
