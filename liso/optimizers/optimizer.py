#!/usr/bin/env python
"""
Optimizer class.

Author: Jun Zhu
"""
from abc import ABC, abstractmethod
import time


class Optimizer(ABC):
    """Abstract Class for Optimizer Object"""
    category = None  # should be either 'global' or 'local'.

    def __init__(self, name):
        """Optimizer Class Initialization

        :param name: string
            Optimizer name.
        """
        self.name = name

        # Random Number Seed (None - Auto-Seed based on time clock)
        self.seed = None
        if self.seed is None:
            self.seed = int(time.time())

        self.printout = 0  # Level of printout

    @abstractmethod
    def __call__(self, opt_problem):
        """Run Optimizer (Calling Routine)

        :param opt_problem: Optimization object
            Optimization problem instance.
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
