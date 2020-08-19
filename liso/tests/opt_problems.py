"""
Optimization problems for unittests of different optimizers.

Author: Jun Zhu, zhujun981661@gmail.com
"""
from abc import ABC, abstractmethod
import numpy as np


class OptimizationTestProblem(ABC):
    @abstractmethod
    def __call__(self, x):
        pass


class EggHolder(OptimizationTestProblem):
    """Egg-holder function.

    https://www.sfu.ca/~ssurjano/egg.html
    f* = -959.6407, x* = (512, 404.2319)
    """
    name = 'Eggholder'
    opt_f = -959.6407
    opt_x = [512, 404.2319]
    x_min = [0, 0]
    x_max = [512, 512]
    n_cons = 0
    n_eq_cons = 0

    def __call__(self, x):
        f = -(x[1] + 47) * np.sin(np.sqrt(abs(x[0] / 2.0 + x[1] + 47.0))) - \
            x[0] * np.sin(np.sqrt(abs(x[0] - x[1] - 47.0)))
        g = [] * self.n_cons
        return [f], g


class Rastrigin(OptimizationTestProblem):
    """2-dimensional Rastrigin function.

    https://www.sfu.ca/~ssurjano/rastr.html
    f* = 0, x* = (0, 0)
    """
    name = 'Rastrigin'
    opt_f = 0
    opt_x = [0, 0]
    x_min = [-5.12, -5.12]
    x_max = [5.12, 5.12]
    n_cons = 0
    n_eq_cons = 0

    def __call__(self, x):
        f = 20 + x[0]**2 + x[1]**2 - 10*np.cos(2*np.pi*x[0]) - 10*np.cos(2*np.pi*x[1])
        g = [] * self.n_cons
        return [f], g


class Rosenbrock(OptimizationTestProblem):
    """Rosenbrock function (Valley-shaped)

    https://www.sfu.ca/~ssurjano/rosen.html
    f* = 0, x* = (1, 1)
    """
    name = 'Rosenbrock'
    opt_f = 0
    opt_x = [1, 1]
    x_min = [-10, -10]
    x_max = [10, 10]
    n_cons = 0
    n_eq_cons = 0

    def __call__(self, x):
        f = (x[0] - 1)**2 + 100*(x[1] - x[0]**2)**2
        g = [] * self.n_cons
        return [f], g


class TP08(OptimizationTestProblem):
    """TP08 constrained Problem.

    min     0.01*x1^2 + x2^2
    s.t.:   25. - x1*x2 <= 0
            25 - x1**2 - x2**2 <= 0
            2 <= x1 <= 50
            0 <= x2 <= 50
    f* = 5, x* = [15.8114, 1.5811]
    """
    name = 'TP08'
    opt_f = 5
    opt_x = [15.8114, 1.5811]
    x_min = [2, 0]
    x_max = [50, 50]
    n_cons = 2
    n_eq_cons = 0

    def __call__(self, x):
        f = 0.01 * x[0] ** 2 + x[1] ** 2
        g = [0.0] * self.n_cons
        g[0] = 25. - x[0] * x[1]
        g[1] = 25. - x[0] ** 2 - x[1] ** 2
        return [f], g


class TP14(OptimizationTestProblem):
    """TP14 constrained Problem.

    min    (x1 - 2)^2 + (x2 - 1)^2
    s.t.:   x1 - 2*x2 + 1 = 0
            -1 + 0.25*x1^2 + x2^2 <= 0
    f* = 1.3935, x* = [0.8229, 0.9114]
    """
    name = 'TP14'
    opt_f = 1.3935
    opt_x = [0.8229, 0.9114]
    x_min = [-10, -10]
    x_max = [10, 10]
    n_cons = 2
    n_eq_cons = 1

    def __call__(self, x):
        f = (x[0] - 2) ** 2 + (x[1] - 1) ** 2
        g = [0.0] * self.n_cons
        g[0] = x[0] - 2.*x[1] + 1
        g[1] = -1 + 0.25*x[0]**2 + x[1]**2
        return [f], g


class TP32(OptimizationTestProblem):
    """TP32 constrained problem (Evtushenko).

    min 	(x1 + 3*x2 + x3)^2 + 4*(x1 - x2)^2
    s.t.:	x1 + x2 + x3 - 1 = 0
            3 + x1^3 - 6*x2 - 4*x3 <= 0
            0 <= xi,  i = 1,2,3
    f* = 1 , x* = [0, 0, 1]
    """
    name = 'TP32'
    opt_f = 1
    opt_x = [0, 0, 1]
    x_min = [0, 0, 0]
    x_max = [20, 20, 20]
    n_cons = 2
    n_eq_cons = 1

    def __call__(self, x):
        f = (x[0] + 3.*x[1] + x[2])**2 + 4.*(x[0] - x[1])**2
        g = [0.0] * self.n_cons
        g[0] = x[0] + x[1] + x[2] - 1.
        g[1] = 3. + x[0]**3 - 6*x[1] - 4*x[2]
        return [f], g


class TP37(OptimizationTestProblem):
    """TP37 constrained problem.

    min 	-x1*x2*x3
    s.t.:	x1 + 2.*x2 + 2.*x3 - 72 <= 0
            - x1 - 2.*x2 - 2.*x3 <= 0
            0 <= xi <= 42,  i = 1,2,3
    f* = -3456 , x* = [24, 12, 12]
    """
    name = 'TP37'
    opt_f = -3456
    opt_x = [24, 12, 12]
    x_min = [0, 0, 0]
    x_max = [42, 42, 42]
    n_cons = 2
    n_eq_cons = 0

    def __call__(self, x):
        f = -x[0] * x[1] * x[2]
        g = [0.0] * self.n_cons
        g[0] = x[0] + 2. * x[1] + 2. * x[2] - 72.0
        g[1] = -x[0] - 2. * x[1] - 2. * x[2]
        return [f], g


class TP43(OptimizationTestProblem):
    """TP43 constrained problem (Rosen-Suzuki).

    min 	x1**2 + x2**2 + 2*x3**2 + x4**2 - 5*x1 - 5*x2 -21*x3 + 7*x4
    s.t.:	x1**2 + x2**2 + x3**2 + x4**2 + x1 - x2 + x3 - x4 - 8 <= 0
            x1**2 + x2**2 + x3**2 + x4**2 - x1 - x4 - 10 <=0
            2*x1**2 + x2**2 + x3**2 + 2*x1 - x2 - x4 - 5 <=0
    f* = -44 , x* = [0, 1, 2, -1]
    """
    name = 'TP43'
    opt_f = -44
    opt_x = [0, 1, 2, -1]
    x_min = [-20, -20, -20, -20]
    x_max = [20, 20, 20, 20]
    n_cons = 3
    n_eq_cons = 0

    def __call__(self, x):
        f = x[0]**2 + x[1]**2 + 2.*x[2]**2 + x[3]**2 - 5.*x[0] - 5.*x[1] - 21.*x[2] + 7.0*x[3]
        g = [0.0] * self.n_cons
        g[0] = x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[0] - x[1] + x[2] - x[3] - 8.0
        g[1] = x[0]**2 + 2.*x[1]**2 + x[2]**2 + 2.*x[3]**2 - x[0] - x[3] - 10.0
        g[2] = 2.*x[0]**2 + x[1]**2 + x[2]**2 + 2.*x[0] - x[1] - x[3] - 5.0
        return [f], g
