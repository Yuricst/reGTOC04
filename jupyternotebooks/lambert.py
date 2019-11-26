"""Lambert problem solution according to Bate's formulation"""
# Python implementation by Yuri Shimane


import numpy as np
import logging
import scipy.optimize as opt
from numpy.linalg import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


#levels: debug, info, warning, error, critical

# define functions that will be used repeatedly in iterative solving step
def _Stumpff_S(z):
    """
    Stumpff function S(z)
    Args:
        z (float): universal anomaly^2/semi-major axis of transfer trajectory
    Returns:
        (float): value of Stumpff functio S(z) evaluated for input z
    """
    if z > 0:
        S = (np.sqrt(z) - np.sin(np.sqrt(z)))/np.power(z,1.5)
    elif z == 0:
        S = 1/6
    elif z < 0:
        S = (np.sinh(np.sqrt(-z)) - np.sqrt(-z))/np.power(-z,1.5)

    return S


def _Stumpff_C(z):
    """
    Stumpff function C(z)
    Args:
        z (float): universal anomaly^2/semi-major axis of transfer trajectory
    Returns:
        (float): value of Stumpff functio S(z) evaluated for input z
    """
    if z > 0:
        C = (1 - np.cos(np.sqrt(z)))/z
    elif z == 0:
        C = 1/2
    elif z < 0:
        C = (np.cosh(np.sqrt(-z)) - 1)/(-z)

    return C

def _y_538(r1,r2,A,z):
    """
    Intermediate function in Lambert problem derivation (eq.5.38 in Curtis)
    Args:
        r1 (1x3 numpy array): position vector of departure
        r2 (1x3 numpy array): position vector of arrival
        A (float): intermediate value related only to input parameters
        z (float): universal anomaly^2/semi-major axis of transfer trajectory
    Returns:
        (float): value of function evaluated for input z
    """

    y = norm(r1) + norm(r2) + A*(z*_Stumpff_S(z) - 1)/np.sqrt(_Stumpff_C(z))

    return y

# Lambert solver
def lambert(mu, r1, r2, tof, grade='pro', method=None, **kwargs):
    """
    Function takes in classic parameters to Lambert problem to determine orbitalelements
    Args:
        r1 (1x3 numpy array): initial position vector of departure [km]
        r2 (1x3 numpy array): final position vector of arrival [km]
        tof (float): time of flight [s]
        mu (float): gravitational parameter [km^3/s^2]
        grade (str): trajectory orientation ('pro' for prograde or 'retro' for retrograde)
        **kwargs: other arguments for scipy.optimize.root_scalar
    Returns:
        (tuple): velocity vector at position 1 and 2 of solution trajectory to Lambert problem
    """
    # boundary values of problem
    logging.info('========================= LAMBERT\'S PROBLEM =========================')
    logging.info(f'Transfering from r1: {r1} [km]')
    logging.info(f'              to r2: {r2} [km]')
    logging.info(f'  in time of flight: {tof/(60*60*24)} [days]')
    logging.info('=====================================================================')

    # compute dtheta [rad]
    dtheta = np.arccos( np.dot(r1,r2)/(norm(r1)*norm(r2)) )
    c12 = np.cross(r1,r2)

    # update dtheta for special cases
    if grade=='retro':
        if c12[2] >= 0:
            dtheta = 2*np.pi - dtheta
    else:
        if c12[2] <= 0:
            dtheta = 2*np.pi - dtheta

    logging.info('dtheta: {}'.format(dtheta*180/(np.pi)))

    # compute input parameter A where A = sin(dtheta) * sqrt[r1*r2 / (1 - cos(dtheta))]
    A = np.sin(dtheta) * np.sqrt(norm(r1)*norm(r2)/(1 - np.cos(dtheta)))
    logging.debug(f'Value of A: {A}')

    # Scipy - NR method scipy.optimize.rootscalar
    def residue_Fz(z,r1,r2,A):
        """Function computes residue of F(z) as defined by Curtis eq. (5.40)
        Args:
            z (float): value of z at which function F is evaluated
            r1 (1x3 numpy array): initial position vector of departure [km]
            r2 (1x3 numpy array): final position vector of arrival [km]
            A (float): variable A predefined in derivation of solution to Lambert's problem
        Returns:
            (tuple): tuple containing residue of F computed at z and Fdot eq. (5.43)
        """
        residue = np.power(_y_538(r1,r2,A,z)/_Stumpff_C(z), 3/2) * _Stumpff_S(z) + A*np.sqrt(_y_538(r1,r2,A,z)) - np.sqrt(mu)*tof

        if z == 0:
            Fdot = np.sqrt(2) * np.power(_y_538(r1,r2,A,0),1.5)/40 + (A/8)*(np.sqrt(_y_538(r1,r2,A,0)) + A*np.sqrt(1/(2*_y_538(r1,r2,A,0))))
        else:
            Fdot = np.power(_y_538(r1,r2,A,z)/_Stumpff_C(z), 1.5) * (((1/(2*z)) * (_Stumpff_C(z) - 3*_Stumpff_S(z)/(2*_Stumpff_C(z)))) + 3*np.power(_Stumpff_S(z),2)/(4*_Stumpff_C(z))) + (A/8)*(3*_Stumpff_S(z)*np.sqrt(_y_538(r1,r2,A,z))/_Stumpff_C(z) + A*np.sqrt(_Stumpff_C(z)/_y_538(r1,r2,A,z)))

        return residue, Fdot

    # Scipy - prepare initial conditions to solve iteratively
    # FIXME - if orbit is retrograde, override bracket_window to only have positive z-values
    #bracket_window = (0.1,5000)
    # FIXME - currently, most robust way is to have a relatively good initial guess...
    z0 = -10
    F0, Fdot0 = residue_Fz(z0,r1,r2,A)
    logging.debug(f'Initially pre-selected z0: {z0}, where F0 = {F0}')
    # if value of F is too big and is NaN, reduce z0 until it isn't
    if np.isnan(F0) == True:
        while np.isnan(F0) == True:
            z0 = z0*0.5
            F0, Fdot0 = residue_Fz(z0,r1,r2,A)
            logging.debug(f'Updated z0: {z0}, where F0 = {F0}')

    if F0 < 0:
        while F0 < 0:
            z0 = z0 + 1
            F0, Fdot0 = residue_Fz(z0,r1,r2,A)

    logging.debug(f'Scipy will use {z0} as z0 initial guess')

    # Scipy - solve to find z-value
    #sol = opt.root_scalar(residue_Fz, args=(r1,r2,A), fprime=True, bracket=bracket_window, method=method, **kwargs)
    sol = opt.root_scalar(residue_Fz, args=(r1,r2,A), fprime=True, x0=z0, method=method, **kwargs)

    if sol.converged:
        z1 = sol.root
    else:
        raise RuntimeError(f'F(z) = 0 calculation failed with initial guess of z {z0}')  # FIXME - document failure


    # display orbit type
    if z1 > 0:
        logging.info(f'Transfer trajectory is an ellipse; z = {z1}')
    elif z1 == 0:
        logging.info(f'Transfer trajectory is a parabola; z = {z1}')
    elif z1 < 0:
        logging.info(f'Transfer trajectory is a hyperbolla; z = {z1}')

    # calculate Lagrange functions
    f    = 1 - _y_538(r1,r2,A,z1)/norm(r1)
    g    = A*np.sqrt(_y_538(r1,r2,A,z1)/mu)
    gdot = 1 - _y_538(r1,r2,A,z1)/norm(r2)
    logging.debug(f'Lagrange functions f: {f}, g: {g}, gdot: {gdot}')

    # calculate initial and final velocity vectors
    v1 = (1/g)*(r2 - f*r1)
    v2 = (1/g)*(gdot*r2 - r1)
    logging.info('=========================== SOLUTION ===========================')
    logging.info(f'Velocity at r1: {v1} [km/s]')
    logging.info(f'velocity at r2: {v2} [km/s]')
    logging.info('================================================================')
    return v1, v2
    


def sv2el(r,v,mu):
    """function converts state-vector to orbital elements
        Args:
        r (lst): position vector [km]
        v (lst): velocity vector [km]
        mu (float): gravitational parameter [km^3/s^2]
    Returns:
        (tuple): tuple of orbital elements
    """
    # specific angular momentum vector
    h_vect = np.cross(r,v)
    h = norm(h_vect)  # scalar quantity
    
    # inclination vector
    i = np.arccos(h_vect[2]/h)

    # eccentricity vector
    e_vect = np.cross(v,h_vect)/mu - r/norm(r)
    e = norm(e_vect)

    # semi-major axis (negative for hyperbolic trajectory)    
    a = h**2/(mu*(1-e**2))

    # right-ascension
    K = [0,0,1] # define K-axis
    N = np.cross(K,h_vect)
    if N[1] > 0:
        RAAN = np.arccos(N[0]/norm(N)) # [rad]
    else:
        RAAN = 2*np.pi - np.arccos(N[0]/norm(N)) # [rad]

    # argument of periapsis
    if e_vect[2] > 0:
        omega = np.arccos(np.dot(e_vect,N)/(e*norm(N)))
    else:
        omega = 2*np.pi - np.arccos(np.dot(e_vect,N)/(e*norm(N)))

    # true anomaly
    v_radial = np.dot(v,r)/norm(r)
    if v_radial > 0:
        theta = np.arccos(np.dot(e_vect,r)/(e*norm(r)))
    else:
        theta = 2*np.pi - np.arccos(np.dot(e_vect,r)/(e*norm(r)))

    # prepare output
    elements = {'i':i, 'h':h, 'e':e,'a':a,'RAAN':RAAN,'omega':omega,'theta':theta}
    return elements


def plot_transfer(r1,v1,r2,v2,mu):
    """Function plots transfer orbits resulting from solving Lambert's problem
    Args:
        r1 (lst): initial position vector [km]
        v1 (lst): initial velocity vector [km]
        r2 (lst): final position vector [km/s]
        v2 (lst): final velocity vector [km/s]
        mu (float): gravitational parameter [km^3/s^2]
    Returns:
        (tuple): tupple of position vector and a 3D plot of transfer
    """
    # fetch orbital elements at initial and final positions
    el1 = sv2el(r1,v1,mu)
    el2 = sv2el(r2,v2,mu)

    # define plotting step-size
    n = 200

    # create array of theta
    if el2.get('theta') > el1.get('theta'):
        theta_range = np.linspace(el1.get('theta'),el2.get('theta'),n)
    else:
        theta_range = np.linspace(el2.get('theta'),el1.get('theta'),n)

    # initialize position vector arrays
    r_PF = np.zeros((3,n))
    r_GEC = np.zeros((3,n))
    # compute perifocal and GEC positions
    for i in range(n):
        tmpPF1 = el1.get('h')**2/(mu*(1 + el1.get('e')*np.cos(theta_range[i]))) * np.array([[np.cos(theta_range[i]), np.sin(theta_range[i]), 0]])
        tmpPF1 = np.transpose(tmpPF1)
        tmpPF2 = np.dot(rotMat(-el2.get('omega'),3), tmpPF1)
        tmpPF3 = np.dot(rotMat(-el2.get('i'),1),  tmpPF2)
        tmpGEC = np.dot(rotMat(-el2.get('RAAN'),3), tmpPF3)

        # store computed values
        for j in range(3):
            r_PF[j,i]  = tmpPF1[j]
            r_GEC[j,i] = tmpGEC[j]

    # plot in perifocal frame
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot()
    plt.plot(r_PF[0,:], r_PF[1,:],'-')
    if el2.get('theta') > el1.get('theta'):
        plt.plot(r_PF[0,0],r_PF[1,0],'xb', label='initial position')
        plt.plot(r_PF[0,n-1],r_PF[1,n-1],'^r', label='final position')
    else:
        plt.plot(r_PF[0,n-1],r_PF[1,n-1],'xb', label='initial position')
        plt.plot(r_PF[0,0],r_PF[1,0],'^r', label='final position')
    plt.title('Lambert transfer in perifocal frame')
    plt.xlabel('x [km]')
    plt.ylabel('y [km]')
    plt.legend()
    plt.show()

    # plot in GEC frame
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(r_GEC[0,:],r_GEC[1,:],r_GEC[2,:]) #,'-')#, c='b', label='Earth',s=16)
    if el2.get('theta') > el1.get('theta'):
        ax.scatter(r_GEC[0,0],r_GEC[1,0],r_GEC[2,0],c='b',m='x', label='initial position')
        ax.scatter(r_GEC[0,n-1],r_GEC[1,n-1],r_GEC[2,n-1],c='r', label='final position')
    else:
        ax.scatter(r_GEC[0,n-1],r_GEC[1,n-1],r_GEC[2,n-1],c='b', label='initial position')
        ax.scatter(r_GEC[0,0],r_GEC[1,0],r_GEC[2,0],c='r', label='final position')
    plt.title('Lambert transfer in center-body-intertial frame')
    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')
    ax.set_zlabel('z [km]')
    ax.legend()
    plt.show()


    return r_PF, r_GEC


def rotMat(phi,axis):
    """Rotational matrix
    Args:
        phi (float): rotation angle in radians
        axis (int): rotation axis 1, 2, or 3
    Returns:
        (matrix): 3x3 rotational matrix
    """
    if axis == 1:
        rotM = np.array([[1, 0, 0],
            [0, np.cos(phi), np.sin(phi)],
            [0, -np.sin(phi),np.cos(phi)]]) 
    elif axis == 2:
        rotM = np.array([[np.cos(phi), 0, -np.sin(phi)],
            [0, 1, 0],
            [np.sin(phi), 0, np.cos(phi)]]) 
    elif axis == 3:
        rotM = np.array([[np.cos(phi), np.sin(phi), 0],
            [-np.sin(phi),np.cos(phi), 0],
            [0, 0, 1]]) 
    else:
        raise RuntimeError('rotation matrix axis must be 1, 2, or 3')

    return rotM

