#!/usr/bin/env python
"""
Optimizer class.

Author: Jun Zhu
"""
from abc import ABC, abstractmethod
import time


class Optimizer(ABC):
    """Abstract class for optimizers.

    Attributes:
        name (str): Name of the optimizer.
        seed (int): Seed for random number. Default = None.
        printout (int): Level of printout.
    """
    category = None  # should be either 'global' or 'local'.

    def __init__(self, name):
        """Optimizer Class Initialization

        :param str name: Optimizer name.
        """
        self.name = name

        self.seed = None
        if self.seed is None:
            self.seed = int(time.time())

        self.printout = 0

    @abstractmethod
    def __call__(self, opt_problem):
        """Run Optimizer (Calling Routine)

        :param Optimization opt_problem: Optimization instance.

        :return: (optimized f,
                  optimized x,
                  miscellaneous information ready for printout).
        :rtype: (float, array-like, str)
        """
        raise NotImplemented

    @abstractmethod
    def __str__(self):
        pass

    @staticmethod
    def _print_title(opt_prob_name):
        print("\n\nSolution for optimization problem '%s' using:" % opt_prob_name)
        print("=" * 80)
        print()

    @staticmethod
    def _print_additional_info(info_list):
        print("\nAdditional information:")
        for info in info_list:
            print('- ' + info)
        print()
