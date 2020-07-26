"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""

# Pure Python/Numpy implementation of the Nelder-Mead algorithm described in
#
# http://www.scholarpedia.org/article/Nelder-Mead_algorithm
#
# Other references:
# - https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method

import numpy as np


def _compute_centroid(simplex):
    """calculate the centroid of all points except the last one.

    :param simplex: list
        A list of (x, f), where x is a 1D numpy.array.
    """
    return np.sum([ele[0] for ele in simplex[:-1]], axis=0) / (len(simplex) - 1)


def _compute_reflection(x, xc, alpha):
    """Get the point in the extended line of x->xc"""
    return xc + alpha * (xc - x)


def _compute_expansion(x, xc, gamma):
    """Get the point in the extended line of xc->x"""
    return xc + gamma * (x - xc)


def _compute_shrink(simplex, sigma):
    """Get the vertices of the shrink simplex."""
    pivot = simplex[0][0]
    return [pivot + sigma*(ele[0] - pivot) for ele in simplex]


def nelder_mead(x0, rtol, atol, max_stag, max_iter, alpha, gamma, beta, sigma, f_obj):
    """Nelder Mead optimizer
    
    :param x0: numpy.ndarray
        Initial vertices of the simplex. It has the shape (num_vertices, num_vars).
    :param rtol: float
        Relative tolerance for objective function.
    :param atol: float
        Absolution tolerance for objective function.
    :param max_stag: int
        Maximum number of stagnated iteration (no improvement).
    :param max_iter: int
        Maximum of iterations.
    :param alpha: float
        Reflection coefficient.
    :param gamma: float
        Expansion coefficient.
    :param beta: float
        Contraction coefficient.
    :param sigma: float
        Shrink coefficient.
    :param f_obj: function object.
        Take the normalized input parameters and output objective value.

    :return: tuple (best parameter array, best score)
    """
    # -----------------------------------------------------------------
    # Check input parameters
    # -----------------------------------------------------------------
    if atol <= 0 or rtol <=0:
        raise ValueError("Tolerance must be positive!")

    if x0.shape[0] != len(x0[0, :]) + 1:
        raise ValueError("Number of vertices of the simplex should be dimension"
                         " of the problem plus 1!")
    # -----------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------
    nfeval = 0  # Number of objective function evaluations
    k_reflection = 0  # Number of reflection operation
    k_expansion = 0  # Number of expansion operation
    k_in_contraction = 0  # Number of inner contraction operation
    k_out_contraction = 0  # Number of outer contraction operation
    k_shrink = 0  # Number of shrink operation

    n_vertex = len(x0) + 1
    simplex = [(x, f_obj(x)) for x in x0]
    nfeval += n_vertex

    simplex.sort(key=lambda x: x[1])
    best_x, best_f = simplex[0]
    best_x_old, best_f_old = best_x, best_f

    stop_info = ""

    # -----------------------------------------------------------------
    # Iterations
    # -----------------------------------------------------------------
    k_failure = 0  # consecutive failure count
    k_iter = 0
    while k_iter < max_iter:
        k_iter += 1

        xc = _compute_centroid(simplex)
        worst_x = simplex[-1][0]
        worst_f = simplex[-1][1]
        second_worst_f = simplex[-2][1]

        reflection_x = _compute_reflection(worst_x, xc, alpha)
        reflection_x = np.maximum(np.minimum(reflection_x, 1), 0)
        reflection_f = f_obj(reflection_x)
        nfeval += 1

        if best_f <= reflection_f < second_worst_f:
            # reflection
            del simplex[-1]
            simplex.append((reflection_x, reflection_f))
            k_reflection += 1

        elif reflection_f < best_f:
            # expansion
            expansion_x = _compute_expansion(reflection_x, xc, gamma)
            expansion_x = np.maximum(np.minimum(expansion_x, 1), 0)
            expansion_f = f_obj(expansion_x)
            nfeval += 1

            del simplex[-1]

            if expansion_f < reflection_f:
                simplex.append((expansion_x, expansion_f))
                k_expansion += 1
            else:
                simplex.append((reflection_x, reflection_f))
                k_reflection += 1

        else:  # reflection_f >= second worst
            do_shrink = True
            if reflection_f < worst_f:
                # outer contraction
                contraction_x = _compute_expansion(reflection_x, xc, beta)
                contraction_x = np.maximum(np.minimum(contraction_x, 1), 0)
                contraction_f = f_obj(contraction_x)
                nfeval += 1

                if contraction_f <= reflection_f:
                    del simplex[-1]
                    simplex.append((contraction_x, contraction_f))
                    do_shrink = False
                    k_out_contraction += 1

            else:  # reflection_f >= worst_f
                # inner contraction
                contraction_x = _compute_expansion(worst_x, xc, beta)
                # Not necessary for inner contraction
                # contraction_x = np.maximum(np.minimum(contraction_x, 1), 0)
                contraction_f = f_obj(contraction_x)
                nfeval += 1

                if contraction_f < worst_f:
                    del simplex[-1]
                    simplex.append((contraction_x, contraction_f))
                    do_shrink = False
                    k_in_contraction += 1

            if do_shrink is True:
                # shrink
                # Do not need to check the boundary
                k_shrink += 1
                new_vertices = _compute_shrink(simplex, sigma)
                simplex = [(best_x, best_f)]
                for i, shrink_x in enumerate(new_vertices):
                    if i > 0:
                        shrink_f = f_obj(shrink_x)
                        nfeval += 1
                        simplex.append((shrink_x, shrink_f))

        # -----------------------------------------------------------------
        # check convergence
        # -----------------------------------------------------------------

        simplex.sort(key=lambda x: x[1])
        best_x, best_f = simplex[0]

        # if objective is too small, use atol
        cdt1 = (abs(best_f_old) < atol / rtol) and (best_f_old - best_f > atol)
        # if objective is big enough, use rtol
        cdt2 = (abs(best_f_old) >= atol / rtol) and best_f_old - best_f > abs(best_f_old * rtol)

        if cdt1 or cdt2:
            # Have significant improvement
            k_failure = 0
            best_f_old = best_f
        else:
            k_failure += 1
            if k_failure > max_stag:
                if abs(best_f_old) < atol / rtol:
                    stop_info = "absolute change of objective function < %f" % atol
                else:
                    stop_info = "relative change of objective function < %f" % rtol
                break

    if k_iter == max_iter:
        stop_info = "maximum number of iteration reached!"

    k_misc = [k_reflection, k_expansion, k_in_contraction, k_out_contraction, k_shrink]
    if sum(k_misc) != k_iter:
        raise SystemError("Unexpected error in function nelder_mead()")

    return best_x, best_f, k_iter, nfeval, k_misc, stop_info
