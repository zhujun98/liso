"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import numpy as np
import h5py

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator

from tests import *


def animate_pso(TestProbCls, filename='/tmp/optimization_history.hdf5', interval=300):
    """Animate a PSO search.

    :param TestProbCls: OptimizationProblem object
        Optimization problem.
    :param filename: string
        Path of the HDF5 file.
    :param interval: int
        Interval between frames (in ms).
    """
    if len(TestProbCls.opt_x) > 2:
        print("Animation is not supported for problems with more "
              "than 2 dimensions!")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_aspect('equal')
    ax.set_title(TestProbCls.name, fontsize=18, y=1.05)

    # Plot the problem function
    x1 = np.linspace(0, 1, 200)
    x2 = np.linspace(0, 1, 200)
    X1, X2 = np.meshgrid(x1, x2)
    test_prob = TestProbCls()
    X1_org = X1 * (test_prob.x_max[0] - test_prob.x_min[0]) + test_prob.x_min[0]
    X2_org = X2 * (test_prob.x_max[1] - test_prob.x_min[1]) + test_prob.x_min[1]
    Z_f, Z_g = test_prob([X1_org, X2_org])

    # Plot the contour for the objective function
    # flip y axis for imshow()
    plt.imshow(Z_f, extent=(0, 1, 1, 0), interpolation='bilinear', cmap='viridis', alpha=0.7)
    plt.contour(X1, X2, Z_f, 10)

    # Draw the lines for the boundaries of constraints
    for i, ele in enumerate(Z_g):
        if i < test_prob.n_eq_cons:
            # each equality constraint is represented by a line
            CS = plt.contour(X1, X2, ele <= 0, 1, cmap='cool')
            plt.setp(CS.collections[0], linewidth=2)
        else:
            # only the intersection of all inequality constraints is shown
            if i == test_prob.n_eq_cons:
                ie_boundary = np.ones_like(ele, dtype=bool)
            ie_boundary = np.logical_and(ie_boundary, ele <= 0)
            if i == test_prob.n_cons - 1:
                CS = plt.contour(X1, X2, ie_boundary, 1, cmap='Greys')
                plt.setp(CS.collections[0], linewidth=2)

    # Mark the ground truth solution as a red star
    ax.plot((test_prob.opt_x[0] - test_prob.x_min[0]) / (test_prob.x_max[0] - test_prob.x_min[0]),
            (test_prob.opt_x[1] - test_prob.x_min[1]) / (test_prob.x_max[1] - test_prob.x_min[1]),
            'r*', ms=25)

    # Animations
    lines = list()
    # a line between the current solution and the last swarm best
    lines.append(ax.plot([], [], 'b-', lw=4)[0])
    # current solution
    lines.append(ax.plot([], [], 'b*', ms=25)[0])
    # particles' best
    lines.append(ax.plot([], [], 'm.', ms=15, alpha=0.7)[0])
    # all particles
    lines.append(ax.plot([], [], 'k.', ms=15)[0])

    fp = h5py.File(filename, 'r')
    x_k = fp['x']
    gbest_x = fp['gbest_x']
    gbest_i = fp['gbest_i']
    pbest_x = fp['pbest_x']

    def update(i):
        lines[0].set_data([gbest_x[i, 0], x_k[i, gbest_i[i], 0]],
                          [gbest_x[i, 1], x_k[i, gbest_i[i], 1]])
        lines[1].set_data(gbest_x[i, 0], gbest_x[i, 1])
        lines[2].set_data(pbest_x[i, :, 0], pbest_x[i, :, 1])
        lines[3].set_data(x_k[i, :, 0], x_k[i, :, 1])
        return lines

    animation = FuncAnimation(fig, update, frames=x_k.shape[0], blit=True,
                              interval=interval, repeat=False)
    plt.show()

    fp.close()


def setup_plot(ax, y, *, x_label='No. of iterations', y_label=None):
    """Plot the evolution of one type of result."""
    if len(y.shape) > 1:
        for i in range(y.shape[1]):
            ax.plot(y[:, i])
    else:
        ax.plot(y)
    ax.set_xlabel(x_label, fontsize=18)
    ax.set_ylabel(y_label, fontsize=18)
    ax.tick_params(labelsize=16)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))


def plot_history(filename='/tmp/optimization_history.hdf5'):
    """Plot evolutions of all important results."""
    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    fp = h5py.File(filename, 'r')

    gbest_L = fp['gbest_L']
    gbest_f = fp['gbest_f']
    gbest_g = fp['gbest_g']
    divergence = fp['divergence']
    w = fp['w']
    lambda_ = fp['lambda']
    rp = fp['rp']
    rho = fp['rho']

    setup_plot(ax[0, 0], gbest_L, y_label="$L$")
    setup_plot(ax[0, 1], gbest_f, y_label="$f$")
    setup_plot(ax[0, 2], gbest_g, y_label="$g$")
    setup_plot(ax[0, 3], divergence, y_label="$\sigma_x$")
    setup_plot(ax[1, 0], w, y_label="$w$", x_label='Outer iterations')
    setup_plot(ax[1, 1], lambda_, y_label='$\lambda_L$', x_label='Outer iterations')
    setup_plot(ax[1, 2], rp, y_label='$r_p$', x_label='Outer iterations')
    setup_plot(ax[1, 3], rho, y_label='$\\rho$')

    plt.suptitle('PSO Optimization Visualization', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    fp.close()


if __name__ == "__main__":
    animate_pso(TP14)
    plot_history()
