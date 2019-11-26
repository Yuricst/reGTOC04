import numpy as np
from numpy import pi, sqrt, cross
from numpy.linalg import norm

from scipy.optimize import newton

# All formulations here are based on
# https://github.com/poliastro/poliastro/blob/PA4817e329a72f849363080c324ac82048c2dfaa/src/poliastro/core/angles.py


def _kepler_eq(EA, MA, ecc):
    return EA - ecc * np.sin(EA) - MA


def _kepler_eq_prime(EA, MA, ecc):
    return 1 - ecc * np.cos(EA)


def _kepler_eq_hyper(HA, MA, ecc):
    return -HA + ecc * np.sinh(HA) - MA


def _kepler_eq_hyper_prime(HA, MA, ecc):
    return ecc * np.cosh(HA) - 1


def _kepler_eq_para(PA, MA, ecc):
    return MA_parabolic(PA, ecc) - MA


def _kepler_eq_para_prime(PA, MA, ecc):
    return MA_parabolic_prime(PA, ecc)


def MA_parabolic(PA, ecc, tol=1e-16):

    x = (ecc-1.0) / (ecc + 1.0) * (PA**2)

    small_term = False
    S = 0.0
    k = 0
    while not small_term:
        term = (ecc - 1.0 / (2.0 * k + 3.0)) * (x**k)
        small_term = np.abs(term) < tol
        S += term
        k += 1

    return (
        np.sqrt(2.0 / (1.0 + ecc)) * PA + np.sqrt(2.0 / (1.0 + ecc) ** 3) * (PA ** 3) * S
    )


def M_parabolic_prime(PA, ecc, tol=1e-16):

    x = (ecc - 1.0) / (ecc + 1.0) * (PA ** 2)

    small_term = False
    S_prime = 0.0
    k = 0

    while not small_term:
        term = (ecc - 1.0 / (2.0 * k + 3.0)) * (2 * k + 3.0) * (x ** k)
        small_term = np.abs(term) < tol
        S_prime += term
        k += 1
    return (
        np.sqrt(2.0 / (1.0 + ecc))
        + np.sqrt(2.0 / (1.0 + ecc) ** 3) * (PA ** 2) * S_prime
    )


def PA2TA(PA):
    """Parabolic Eccentric Anomaly to True Anomaly"""

    return 2.0 * np.arctan(PA)


def TA2PA(TA):
    """True Anomaly to Parabolic Eccentric Anomaly"""

    return np.tan(TA/2.0)


def TA2EA(TA, ecc):
    """True Anomaly to Eccentric Anomaly"""

    beta = ecc / (1 + np.sqrt(1 - ecc**2))

    EA = TA - 2 * np.arctan(beta * np.sin(TA) / (1 + beta * np.cos(TA)))

    return EA


def TA2HA(TA, ecc):
    """True Anomaly to Hyperbolic Eccentric Anomaly"""

    HA = np.log(
        (np.sqrt(ecc + 1) + np.sqrt(ecc - 1) * np.tan(TA / 2))
        / (np.sqrt(ecc + 1) - np.sqrt(ecc - 1) * np.tan(TA / 2))
    )
    return HA


def EA2TA(EA, ecc):
    """Eccentric Anomaly to True Anomaly"""

    beta = ecc / (1 + np.sqrt((1 - ecc) * (1 + ecc)))
    TA = EA + 2 * np.arctan(beta * np.sin(EA) / (1 - beta * np.cos(EA)))
    return TA


def HA2TA(HA, ecc):
    """Hyperbolic eccentric anomaly to true anomaly"""

    TA = 2 * np.arctan(
        (np.exp(HA) * np.sqrt(ecc + 1) - np.sqrt(ecc + 1))
        / (np.exp(HA) * np.sqrt(ecc - 1) + np.sqrt(ecc - 1))
    )
    return TA


def MA2EA(MA, ecc, **kwargs):
    """Mean Anomaly to Eccentric Anomaly"""

    assert ecc <= 1.0, f'Eccentricity must be between 0 and 1 for this function. Currently ecc={ecc}'
    assert ecc >= 0.0, f'Eccentricity must be between 0 and 1 for this function. Currently ecc={ecc}'

    x0 = MA
    try:
        EA = newton(_kepler_eq, x0, args=(MA, ecc), fprime=_kepler_eq_prime, **kwargs)
        return EA
    except Exception as e:
        raise RuntimeError(e)


def MA2HA(MA, ecc, **kwargs):
    """Mean Anomaly to Hyperbolic Eccentric Anomaly"""

    assert ecc >= 1.0, f'Eccentricity must be greater than 1 for this function. Currently ecc={ecc}'

    x0 = np.arcsinh(MA/ecc)
    try:
        HA = newton(_kepler_eq_hyper, x0, args=(MA, ecc), fprime=_kepler_eq_hyper_prime, **kwargs)
        return HA
    except Exception as e:
        raise RuntimeError(e)


def MA2PA(MA, ecc, **kwargs):
    """Mean Anomaly to Parabolic Eccentric Anomaly"""
    assert ecc <= 1.0 + \
        1e-2, f'Eccentricity must be between 0.99 and 1.01 for this function. Currently ecc={ecc}'
    assert ecc >= 1.0 - \
        1e-2, f'Eccentricity must be between 0.99 and 1.01 for this function. Currently ecc={ecc}'

    B = 3.0 * MA / 2.0
    A = (B + (1.0 + B ** 2) ** 0.5) ** (2.0 / 3.0)
    x0 = 2 * A * B / (1 + A + A ** 2)

    try:
        PA = newton(_kepler_eq_para, x0, args=(MA, ecc), fprime=_kepler_eq_para_prime, **kwargs)
        return PA
    except Exception as e:
        raise RuntimeError(e)


def EA2MA(EA, ecc):
    """Eccentric Anomaly to Mean Anomaly"""

    MA = _kepler_eq(EA, 0.0, ecc)
    return MA


def HA2MA(HA, ecc):
    """Hyperbolic Anomaly to Mean Anomaly"""

    MA = _kepler_eq_hyper(HA, 0.0, ecc)

    return MA


def PA2MA(PA, ecc):
    """Parabolic Anomaly to Mean Anomaly"""

    MA = _kepler_eq_para(PA, 0.0, ecc)

    return MA


def MA2TA(MA, ecc, tol=1e-2, **kwargs):
    """Mean Anomaly to True Anomaly"""

    if ecc > (1.0 + tol):
        # hyperbolic
        HA = MA2HA(MA, ecc, **kwargs)
        TA = HA2TA(HA, ecc)

    elif ecc < (1.0 - tol):
        # elliptic
        EA = MA2EA(MA, ecc, **kwargs)
        TA = EA2TA(EA, ecc)

    else:
        # near parabolic
        PA = MA2PA(MA, ecc, **kwargs)  # PA is parabolic eccentric anomaly
        TA = PA2TA(PA)

    return TA


def TA2MA(TA, ecc, tol=1e-2):
    """True Anomaly to Mean Anomaly"""

    if ecc > 1 + tol:
        # hyperbolic
        HA = TA2HA(TA, ecc)
        MA = HA2MA(HA, ecc)

    elif ecc < 1 - tol:
        # elliptic
        EA = TA2EA(TA, ecc)
        MA = EA2MA(EA, ecc)

    else:
        # near parabolic
        PA = TA2PA(TA)  # PA is parabolic eccentric anomaly
        MA = PA2MA(PA, ecc)

    return MA
