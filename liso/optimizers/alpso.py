#!/usr/bin/env python
"""
Single-objective Augmented Lagrangian Particle Swarm Optimizer (ALPSO)

The optimizer solves problems of the form:

    min F(x)

    subject to: G_i(x)  = 0, i = 1, 2, ..., M_E
                G_j(x) <= 0, j = M_E+1, M_E+2, ..., M
                x_LB <= x <= x_UB

Author: Jun Zhu


TODO: Improve convergence condition.
"""
import numpy as np
import h5py

from ..config import Config

INF = Config.INF


def update_inertial_weight(w0, w1, k):
    """Update inertial weight w.

    :param w0: float
        Initial inertial weight.
    :param w1: float
        Final inertial weight.
    :param k: float
        Coefficient.
    """
    w = w1 + (w0 - w1) * k
    w = min(max(w, min(w0, w1)), max(w0, w1))
    return w


def alpso(x0,
          v0,
          n_cons,
          n_eq_cons,
          topology,
          max_outer_iter,
          max_inner_iter,
          min_inner_iter,
          etol,
          itol,
          rtol,
          atol,
          dtol,
          c1,
          c2,
          w0,
          w1,
          use_gcpso,
          nf,
          ns,
          rho_max,
          rho_min,
          f_obj_con):
    """Augmented Lagrangian Particle Swarm Optimizer.

    :param x0: numpy.ndarray (particle ID, position vector)
        Initial positions. It has the shape (swarm_size, num_vars)
    :param v0: numpy.ndarray (particle ID, velocity vector)
        Initial velocities. It has the shape (swarm_size, num_vars)
    :param n_cons: int
        Total number of constraints.
    :param n_eq_cons: int
        Number of equality constraints.
    :param topology: string
        Particle topology.
    :param max_outer_iter: int
        Maximum number of outer iterations.
    :param max_inner_iter: int
        Maximum number of inner iterations.
    :param min_inner_iter: int
        Minimum number of inner iterations.
    :param etol: float
        Absolute tolerance for equality constraints.
    :param itol: float
        Absolute tolerance for inequality constraints.
    :param rtol: float
        Relative tolerance for Lagrange function.
    :param atol: float
        Absolution tolerance for Lagrange function.
    :param dtol: float
        Absolute tolerance for position deviation of all particles.
    :param c1: float
        Cognitive Parameter.
    :param c2: float
        Social Parameter.
    :param w0: float
        Initial inertial weight.
    :param w1: float
        Initial inertial weight.
    :param use_gcpso: bool
        Use GCPSO:
        F. Bergh, A.P. Engelbrecht,
        A new locally convergent particle swarm optimiser,
        Systems, Man and Cybernetics, 2002 IEEE International Conference.
    :param ns: int
        Number of Consecutive Successes in Finding New Best Position
        of Best Particle Before Search Radius will be Increased (GCPSO)
    :param nf: int
        Number of Consecutive Failures in Finding New Best Position
        of Best Particle Before Search Radius will be Increased (GCPSO).
    :param rho_max: float
        Maximum search radius (GCPSO). A large rho_max will affect the
        convergence condition.
    :param rho_min: float
        Minimum search radius (GCPSO).
    :param f_obj_con: function object.
        Take the normalized input parameters and output a tuple of
        (objective, constraints).

    :return:
    """
    # -----------------------------------------------------------------
    # Check input parameters
    # -----------------------------------------------------------------
    if atol <= 0 or rtol <=0 or dtol <=0 or itol <= 0 or etol <= 0:
        raise ValueError("Tolerance must be positive!")

    if x0.max() > 1 or x0.min() < 0:
        raise ValueError("Positions must be normalized in the space [0, 1].")

    if n_cons < 0 or n_eq_cons < 0 or n_cons < n_eq_cons:
        raise ValueError("Wrong number of (equal) constraints!")

    if max_outer_iter < 1 or max_inner_iter < 1 or min_inner_iter < 1:
        raise ValueError("No. of iterators must be at least 1!")

    if max_outer_iter < min_inner_iter:
        raise ValueError("max_outer_iter < min_inner_inter!")

    if topology.lower() not in ['gbest']:
        raise ValueError("Unknown topology!")

    # -----------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------

    # Normalized units are used in the following calculation:
    # - The position space is normalized to [0, 1];
    # - Time step is 1.
    swarm_size, n_vars = x0.shape
    x_k = np.copy(x0)  # position
    divergence = np.sqrt(np.sum(np.var(x_k, axis=0)))  # position divergence
    divergence0 = divergence  # used for inertial weight update
    v_k = np.copy(v0)  # velocity

    lambda_ = np.zeros(n_cons, float)  # Lagrangian multiplier
    rp = np.ones(n_cons, float)  # penalty factor (quadratic term)
    w = w0  # inertial weight
    rho = rho_max  # search radius for the global best particle (GCPSO)
    nfeval = 0  # No. of evaluations of the objective function

    f = np.ones(swarm_size, float) * INF  # objective
    L = np.ones(swarm_size, float) * INF  # Lagrangian function
    g = np.ones([swarm_size, n_cons], float) * INF  # Constraint
    for i in range(swarm_size):
        f[i], g[i, :] = f_obj_con(x_k[i, :])
        nfeval += 1
        L[i] = f[i]
        for j in range(n_cons):
            # Equality Constraints
            if j < n_eq_cons:
                theta = g[i, j]
            # Inequality Constraints
            else:
                theta = max(g[i, j], -lambda_[j] / (2.0 * rp[j]))

            L[i] += lambda_[j]*theta + rp[j]*theta**2

    # global best
    gbest_i = np.argmin(L)
    gbest_x = np.copy(x_k[gbest_i, :])
    gbest_L = L[gbest_i]
    gbest_f = f[gbest_i]
    gbest_g = np.copy(g[gbest_i, :])

    # particles' best
    pbest_x = np.copy(x_k)
    pbest_L = np.copy(L)

    # Initialize history data file
    params_history = dict()
    params_history['gbest_i'] = [gbest_i]
    params_history['x'] = [x_k.tolist()]
    params_history['pbest_x'] = [pbest_x.tolist()]
    params_history['gbest_x'] = [gbest_x.tolist()]
    params_history['gbest_L'] = [gbest_L]
    params_history['gbest_f'] = [gbest_f]
    params_history['gbest_g'] = [gbest_g.tolist()]
    params_history['divergence'] = [divergence]
    params_history['lambda'] = [lambda_.tolist()]
    params_history['rp'] = [rp.tolist()]
    params_history['w'] = [w]
    params_history['rho'] = [rho]

    stop_info = ""

    # -----------------------------------------------------------------
    # Iterations
    # -----------------------------------------------------------------

    # Outer optimization loop
    k_out = 0  # outer loop count
    k_success = 0  # consecutive success count
    k_failure = 0  # consecutive failure count
    while k_out < max_outer_iter:
        k_out += 1

        # -------------------------------------------------------------
        # Each inner loop can be seen as a sub-problem. Lagrangian
        # multiplier, the penalty factor and inertial weight will
        # not be updated in the inner loop
        # -------------------------------------------------------------
        gbest_L_old = np.copy(gbest_L)
        gbest_g_old = np.copy(gbest_g)
        k_inn = 0   # inner loop count
        while k_inn < max_inner_iter:
            k_inn += 1

            # Update position and velocity
            for i in range(swarm_size):
                # position and velocity update
                if use_gcpso is True and i == gbest_i:
                    rr = 1.0 - 2.0 * np.random.rand(n_vars)
                    v_k[i, :] = -x_k[i, :] + gbest_x + w*v_k[i, :] + rr*rho
                else:
                    r1 = np.random.rand(n_vars)
                    r2 = np.random.rand(n_vars)

                    # K. Sedlaczek and P. Eberhard,
                    # Struct. Multidisc. Optim. (2006) 32: 277–286

                    v_k[i, :] = w*v_k[i, :] \
                                + c1*r1*(pbest_x[i, :] - x_k[i, :]) \
                                + c2*r2*(gbest_x - x_k[i, :])

                x_k[i, :] += v_k[i, :]

                # check fitness
                x_k[i, :] = np.maximum(np.minimum(x_k[i, :], 1), 0)

                # update Lagrangian function values
                f[i], g[i, :] = f_obj_con(x_k[i, :])
                nfeval += 1

                L[i] = f[i]
                for j in range(n_cons):
                    # Equality Constraints
                    if j < n_eq_cons:
                        theta = g[i, j]
                    # Inequality Constraints
                    else:
                        theta = max(g[i, j], -lambda_[j] / (2.0 * rp[j]))

                    L[i] += lambda_[j]*theta + rp[j]*theta**2

                # update particle best
                if L[i] < pbest_L[i]:
                    pbest_L[i] = L[i]
                    pbest_x[i, :] = np.copy(x_k[i, :])

                # update global best
                if L[i] < gbest_L:
                    gbest_i = i
                    gbest_x = np.copy(x_k[i, :])
                    gbest_L = L[i]
                    gbest_f = f[i]
                    gbest_g = np.copy(g[i, :])

            # update divergence
            divergence = np.sqrt(np.sum(np.var(x_k, axis=0)))

            # update history
            params_history['x'].append(x_k.tolist())
            params_history['gbest_i'].append(gbest_i)
            params_history['pbest_x'].append(pbest_x.tolist())
            params_history['gbest_x'].append(gbest_x.tolist())
            params_history['gbest_L'].append(gbest_L)
            params_history['gbest_f'].append(gbest_f)
            params_history['gbest_g'].append(gbest_g.tolist())
            params_history['divergence'].append(divergence)
            params_history['rho'].append(rho)

            if use_gcpso is True:
                if gbest_L >= gbest_L_old:
                    k_failure += 1
                    k_success = 0
                else:
                    k_success += 1
                    k_failure = 0

                # Update search radius for the best particle (GCPSO)
                if k_success > ns:
                    rho *= 2.0
                    k_success = 0
                elif k_failure > nf:
                    rho *= 0.5
                    k_failure = 0

                rho = max(min(rho_max, rho), rho_min)

            # check convergence of the inner loop
            if gbest_L < gbest_L_old and k_inn >= min_inner_iter:
                break

        # End of inner loop
        # -------------------------------------------------------------

        # -------------------------------------------------------------
        # Check constraints
        # -------------------------------------------------------------
        constraints_satisfied = True
        for j in range(n_cons):
            if j < n_eq_cons:
                if abs(gbest_g[j]) > etol:
                    constraints_satisfied = False
                    break
            else:
                if gbest_g[j] > itol:
                    constraints_satisfied = False
                    break

        # Check position and objective convergence
        convergence_satisfied = False
        if constraints_satisfied is True and abs(divergence) <= dtol:
            if abs(gbest_L - gbest_L_old) <= rtol*abs(gbest_L_old):
                stop_info = "Relative change of Lagrangian function < %f" % rtol
                convergence_satisfied = True
            elif abs(gbest_L - gbest_L_old) <= atol:
                stop_info = "Absolute change of Lagrangian function < %f" % atol
                convergence_satisfied = True

        if constraints_satisfied is True and convergence_satisfied is True:
            break
        elif k_out == max_outer_iter:
            stop_info = "Maximum number of iteration reached!"
            break
        else:
            # update inertial weight
            w = update_inertial_weight(w0, w1, divergence / divergence0)

            # Update Lagrange multiplier and penalty factor rp
            for j in range(n_cons):
                # Equality constraints
                if j < n_eq_cons:
                    theta = gbest_g[j]
                    if abs(gbest_g[j]) > abs(gbest_g_old[j]) and abs(gbest_g[j]) > etol:
                        rp[j] *= 2.0
                    elif abs(gbest_g[j]) <= etol:
                        rp[j] *= 0.5
                    rp[j] = max(rp[j], 0.5 * (abs(lambda_[j]) / etol) ** 0.5)

                # Inequality constraints
                else:
                    theta = max(gbest_g[j], -lambda_[j] / (2.0 * rp[j]))

                    if gbest_g[j] > gbest_g_old[j] and gbest_g[j] > itol:
                        rp[j] *= 2.0
                    elif gbest_g[j] <= itol:
                        rp[j] *= 0.5
                    rp[j] = max(rp[j], 0.5 * (abs(lambda_[j]) / itol) ** 0.5)

                lambda_[j] += 2.0 * rp[j] * theta

            # update history record only updated every outer loop
            params_history['lambda'].append(lambda_.tolist())
            params_history['rp'].append(rp.tolist())
            params_history['w'].append(w)

            # Update the Lagrangian since the Lagrangian multiplier and the
            # penalty factor could have changed.
            for i in range(swarm_size):
                L[i] = f[i]
                for j in range(n_cons):
                    # Equality Constraints
                    if j < n_eq_cons:
                        theta = g[i, j]
                    # Inequality Constraints
                    else:
                        theta = max(g[i, j], -lambda_[j] / (2.0 * rp[j]))

                    L[i] += lambda_[j] * theta + rp[j] * theta ** 2

            # *********************************************************
            # Important! Since each inner loop is a new unconstrained
            # PSO problem with different objective function (new
            # Lagrangian multiplier and penalty factor).
            # *********************************************************

            # reset swarm's best for the next inner loop
            gbest_i = L.argmin()
            gbest_x = np.copy(x_k[gbest_i, :])
            gbest_L = L[gbest_i]
            gbest_f = f[gbest_i]
            gbest_g = np.copy(g[gbest_i, :])

            # reset particles' best for the next inner loop
            pbest_x = np.copy(x_k)
            pbest_L = np.copy(L)

    # save the optimization history to an hdf5 file
    with h5py.File('/tmp/optimization_history.hdf5', 'w') as fp:
        for key in params_history.keys():
            fp.create_dataset(key, data=np.array(params_history[key]))

    # End of outer loop
    # -------------------------------------------------------------
    return gbest_x, gbest_L, gbest_f, gbest_g, lambda_, rp, k_out, nfeval, stop_info
