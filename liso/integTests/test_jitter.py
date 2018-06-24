#!/usr/bin/python
"""
Unittest of jitter study with ASTRA.

Author: Jun Zhu, zhujun981661@gmail.com
"""
import unittest

from liso import Linac, LinacJitter
from liso.integTests.helpers import print_title


class TestJitter(unittest.TestCase):
    def setUp(self):
        linac = Linac()
        linac.add_beamline('astra',
                           name='gun',
                           fin='integTests/astra_gun/injector.in',
                           template='integTests/jitter/injector.in.000',
                           pout='injector.0150.001')

        print(linac)

        # set an jitter problem
        self.jt = LinacJitter(linac)

        self.jt.add_response('emitx', expr='gun.out.emitx', scale=1e6)
        self.jt.add_response('Ct', expr='gun.out.Ct', scale=1e15)
        self.jt.add_response('gamma', expr='gun.out.gamma')

        self.jt.add_jitter('gun_gradient', value=110, sigma=-0.001)
        self.jt.add_jitter('gun_phase', value=0.0, sigma=0.01)

    def test_not_raise(self):
        print_title("Test jitter simulation with ASTRA!")
        self.jt.monitor_time = True
        self.jt.run(5)


if __name__ == "__main__":
    unittest.main()
