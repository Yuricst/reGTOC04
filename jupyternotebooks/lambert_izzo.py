import numpy as np
from numpy.linalg import norm
from numpy import cross
import logging
from scipy.special import hyp2f1

# Implementation of Izzo's algo for solving lamberts problem

def lambert(mu, r1, r2, tof, M=0, numiter=35, rtol=1e-8, return_="short"):

    '''Solves Lambert's problem

    Args:
        mu (float): gravitional constant km3/s2
        r1 (np.array): initial position vector
        r2 (np.array): final position vector
        tof (float): time of flight (seconds)
        M (int): number of complete revolutions
        numiter (int): maximum number of iterations in solver
        rtol (float): tolerance of solver
        return_ (str): type of solution to be returned (see note)

    Returns:
        if return_ is 'short' or 'long':
            (tuple): (v1, v2), initial and final velocities as np.array
        if return_ is 'all':
            (list): list of all possible initial and final velocities for this trajectory

    '''

    sols = list(izzo(mu, r1, r2, tof, M, numiter, rtol))

    if return_ == 'short':
        return sols[0]
    elif return_ == 'long':
        return sols[-1]
    elif return_ == 'all':
        return sols

    return ValueError("return_ must be either 'short', 'long' or 'all'.")


def izzo(mu, r1, r2, tof, M, numiter, rtol):
    '''Solves lamberts problem using izzo, adapting implementation from poliastro.

    Args:
        mu (float): gravitional constant km3/s2
        r1 (np.array): initial position vector
        r2 (np.array): final position vector
        tof (float): time of flight (seconds)
        M (int): number of complete revolutions
        numiter (int): maximum number of iterations in solver
        rtol (float): tolerance of solver

    Yields:
        (tuple): (v1, v2) initial and final velocities

    '''
    assert tof > 0
    assert mu > 0

    # check collinearity
    if np.all(cross(r1,r2) == 0):
        raise ValueError('Lamberts cant be solved for collinear vectors')

    # chord
    c = r2 - r1
    c_norm = norm(c)
    r1_norm = norm(r1)
    r2_norm = norm(r2)

    # semiperimeter
    s = 0.5*(r1_norm + r2_norm + c_norm)

    # unit vectors
    i_r1, i_r2 = r1/r1_norm, r2/r2_norm
    i_h = cross(i_r1, i_r2)
    i_h = i_h / norm(i_h)


    # geometry
    ll = np.sqrt(1 - min(1.0, c_norm/s))

    if i_h[2] < 0:
        ll = -ll
        i_h = - i_h

    i_t1 = cross(i_h, i_r1)
    i_t2 = cross(i_h, i_r2)

    # Non dimensional time of flight
    T = np.sqrt(2 * mu / s ** 3) * tof

    # Find solutions
    xy = _find_xy(ll, T, M, numiter, rtol)

    # reconstruct solution

    gamma = np.sqrt(mu * s / 2)
    rho = (r1_norm - r2_norm)/c_norm
    sigma = np.sqrt(1- rho**2)

    for x, y in xy:
        V_r1, V_r2, V_t1, V_t2 = _reconstruct(x, y, r1_norm, r2_norm, ll, gamma, rho, sigma)

        v1 = V_r1 * i_r1 + V_t1 * i_t1
        v2 = V_r2 * i_r2 + V_t2 * i_t2

        yield v1, v2




def _find_xy(ll, T, M, numiter, rtol):

    '''Compute all x, y for given revolutions'''

    assert abs(ll) < 1
    assert T > 0

    #compute max complete revolutions
    M_max = np.floor(T/np.pi)

    T_00 = np.arccos(ll) + ll * np.sqrt(1 - ll**2)

    # refine number of revolutions
    if T < T_00 + M_max*np.pi and M_max > 0:
        raise ValueError('No feasible solution, try lower M')

    # initial guess
    for x_0 in _initial_guess(T, ll, M):
        # start householder iterations
        x = _householder(x_0, T, ll, M, rtol, numiter)
        y = _compute_y(x, ll)

        yield x, y


def _reconstruct(x, y, r1, r2, ll, gamma, rho, sigma):
    """
    Reconstruct solution velocity vectors.
    """

    V_r1 = gamma * ((ll * y - x) - rho * (ll * y + x)) / r1
    V_r2 = -gamma * ((ll * y - x) + rho * (ll * y + x)) / r2
    V_t1 = gamma * sigma * (y + ll * x) / r1
    V_t2 = gamma * sigma * (y + ll * x) / r2

    return [V_r1, V_r2, V_t1, V_t2]


def _initial_guess(T, ll, M):

    """
    Initial guess.
    """
    if M == 0:
        # Single revolution
        T_0 = np.arccos(ll) + ll * np.sqrt(1 - ll ** 2) + M * np.pi  # Equation 19
        T_1 = 2 * (1 - ll ** 3) / 3  # Equation 21
        if T >= T_0:
            x_0 = (T_0 / T) ** (2 / 3) - 1
        elif T < T_1:
            x_0 = 5 / 2 * T_1 / T * (T_1 - T) / (1 - ll ** 5) + 1
        else:
            # This is the real condition, which is not exactly equivalent
            # elif T_1 < T < T_0
            x_0 = (T_0 / T) ** (np.log2(T_1 / T_0)) - 1

        return [x_0]
    else:
        # Multiple revolution
        x_0l = (((M * np.pi + np.pi) / (8 * T)) ** (2 / 3) - 1) / (
            ((M * np.pi + np.pi) / (8 * T)) ** (2 / 3) + 1
        )
        x_0r = (((8 * T) / (M * np.pi)) ** (2 / 3) - 1) / (
            ((8 * T) / (M * np.pi)) ** (2 / 3) + 1
        )

        return [x_0l, x_0r]



def _householder(p0, T0, ll, M, tol, maxiter):

    ''' Root finding using householder algo'''

    for ii in range(maxiter):
        y = _compute_y(p0, ll)

        fval = _tof_equation_y(p0, y, T0, ll, M)
        T = fval + T0
        fder, fder2, fder3 = _tof_equation_p(p0, y, T, ll)

        # Householder step (quartic)
        p = p0 - fval * (
            (fder ** 2 - fval * fder2 / 2)
            / (fder * (fder ** 2 - fval * fder2) + fder3 * fval ** 2 / 6)
        )

        if abs(p - p0) < tol:
            return p
        p0 = p

    raise RuntimeError('Failed to converge')

def _tof_equation_y(x, y, T0, ll, M):

    '''calculate time of flight'''

    if M == 0 and np.sqrt(0.6) < x < np.sqrt(1.4):
        eta = y - ll * x
        S_1 = (1- ll - x * eta)**0.5
        a, b, c = 3, 1, 5/2 # parameters for hyp2f1
        Q = 4 / 3 * hyp2f1(a, b, c, S_1)
        T_ = (eta**3 * Q + 4*ll*eta)**0.5
    else:
        psi = _compute_psi(x, y, ll)
        T_ = np.divide(
            np.divide(psi + M*np.pi, np.sqrt(np.abs(1 - x ** 2))) - x + ll * y,
            (1 - x ** 2))

    return T_ - T0

def _compute_psi(x, y, ll):

    """
    Computes psi.

    The auxiliary angle psi is computed using Eq.(17) by the appropriate
    inverse function

    """
    if -1 <= x < 1:
        # Elliptic motion
        # Use arc cosine to avoid numerical errors
        return np.arccos(x * y + ll * (1 - x ** 2))
    elif x > 1:
        # Hyperbolic motion
        # The hyperbolic sine is bijective
        return np.arcsinh((y - x * ll) * np.sqrt(x ** 2 - 1))
    else:
        # Parabolic motion
        return 0.0


def _tof_equation_p(x, y, T, ll):
    '''
    Returns the derivatives of TOF
    '''
    #not sure when x -> 1
    dT =  (3 * T * x - 2 + 2 * ll ** 3 * x / y) / (1 - x ** 2)
    ddT = (3 * T + 5 * x * dT + 2 * (1 - ll ** 2) * ll ** 3 / y ** 3) / (1 - x ** 2)
    dddT = (7 * x * ddT + 8 * dT - 6 * (1 - ll ** 2) * ll ** 5 * x / y ** 5) / (
        1 - x ** 2
    )

    return dT, ddT, dddT


def _compute_y(x, ll):
    """
    Computes y.
    """
    return np.sqrt(1 - ll ** 2 * (1 - x ** 2))
