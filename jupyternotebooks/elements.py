import numpy as np
from numpy import cross, pi, sqrt
from numpy.linalg import norm

def angle_between(a, b):
    """Returns angle (rad) between vectors `a` abd `b`.

    Args:
        a (np.array): Vector `a`.
        b (np.array): Vector `b`.

    Returns:
        float: Angle between `a` and `b` in radians.
    """

    return np.arccos(np.dot(a,b)/norm(a)/norm(b))



def rv2coe(mu, r, v, tol=1e-12):
    """state vector to classical orbital elements, follows derivation by Curtis. 
    Distinction of cases is made for equatorial and circular cases explicitly

    Args:
        mu (float): Gravitational Parameter (km3/s2).
        r (np.array): Position Vector (km).
        v (np.array): Velocity Vector (km/s).
        tol (float): Tolerance. Defaults to 1e-12.

    Returns:
        tuple: Orbital elements tuple:
            (angular momentum magnitude (km2/s),
             eccentricity magnitude,
             longitude of ascending node (rad),
             inclination (rad),
             argument of perigee (rad),
             true anomaly (rad)).

    Examples
        Examples should be written in doctest format, and
        should illustrate how to use the function/class.
        >>>

    """

    vr = np.dot(r, v)/norm(r)

    h = cross(r, v)

    inc = np.arccos(h[2]/norm(h))

    n = cross([0,0,1], h)

    e = (1/mu)*((np.dot(v,v) - mu/norm(r))*r - norm(r)*vr*v)
    ecc = norm(e)


    circular = ecc < tol
    equatorial = abs(inc) < tol

    if not equatorial:
        LAN = np.arccos(n[0]/norm(n))
        if n[1] < 0:
            LAN = 2*pi - LAN
    else:
        LAN = 0;

    if not equatorial and not circular:
        argp = angle_between(n,e)
        if e[2] < 0:
            argp = 2*pi - argp

        trueAnom = angle_between(e,r)
        if vr < 0:
            trueAnom = 2*pi - trueAnom

    elif not equatorial and circular:
        argp = 0

        trueAnom = angle_between(n, r)
        if r[2] < 0:
            trueAnom = 2*pi - trueAnom

    elif equatorial and not circular:
        argp = np.arctan2(e[1],e[0])
        #argp = np.arccos(e[0]/norm(e))
        #if e[1] < 0:
        #    argp = 2*pi - argp

        trueAnom = angle_between(e,r)
        if vr < 0:
            trueAnom = 2*pi - trueAnom

    elif equatorial and circular:
        argp = 0;

        trueAnom = np.arctan2(r[1], r[0])
        #trueAnom = np.arccos(r[0]/norm(r))
        #if r[1] < 0:
        #    trueAnom = 2*pi - trueAnom

    a = (np.dot(h,h)/mu)/(1-np.dot(e,e))

    return norm(h), norm(e), LAN % (2*pi), inc % (2*pi), argp % (2*pi), trueAnom % (2*pi)



def coe2rv(mu, h, e, LAN, inc, argp, trueAnom):
    """Converts classical orbital elements to state vector"""

    # define basis vectors
    u1 = np.array([1,0,0])
    u2 = np.array([0,1,0])
    u3 = np.array([0,0,1])

    rp = (h**2/mu) * (1 / (1 + e*np.cos(trueAnom))) * (np.cos(trueAnom)*u1 + np.sin(trueAnom)*u2)
    vp = (mu/h) * (-np.sin(trueAnom)*u1 + (e + np.cos(trueAnom))*u2)

    cosLAN = np.cos(LAN)
    sinLAN = np.sin(LAN)
    R3_LAN = np.array([[cosLAN, sinLAN, 0], [-sinLAN, cosLAN, 0], [0,0,1]])

    cosinc = np.cos(inc)
    sininc = np.sin(inc)
    R1_inc = np.array([[1, 0, 0], [0, cosinc, sininc], [0, -sininc, cosinc]])

    cosArgp = np.cos(argp)
    sinArgp = np.sin(argp)
    R3_argp = np.array([[cosArgp, sinArgp, 0], [-sinArgp, cosArgp, 0], [0, 0, 1]])

    Q_pX = np.transpose(R3_argp @ R1_inc @ R3_LAN)

    r = Q_pX @ rp
    v = Q_pX @ vp

    return r, v
