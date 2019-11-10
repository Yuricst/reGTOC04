import numpy as np
import scipy as sp
import pandas as pd
import scipy.optimize as opt
from numpy import linalg as linalg
from numpy.linalg import norm

class Satellite:

    def __init__(self):

        self.mu = 1.32712440018e+11  # gravitational parameter of Sun km3/s2
        self.u_AU = 1.495978707e+8   # 1 AU in km
        self.u_day = 86400           # 1 day in seconds
        self.u_deg2rad = np.pi/180   # conversion from degree to radians

        self.r = None
        self.v = None

        self.name = 'Scaramouche'

    def set_state(self, epoch, r, v, add_to_hist=False):

        self.epoch = epoch
        self.r  = r
        self.v  = v

        return None

    def get_state(self):

        return self.epoch, self.r, self.v

    def state_to_orb_elements(self, epoch, r, v):

        #calculate angular momentum
        h = np.cross(r,v)
        print(h)

        #calculate inclination
        i = np.arccos(h[2]/norm(h))

        # eccentricity
        e = np.cross(v, h)/self.mu - r/norm(r)
        e_mag = norm(e)

        # right ascension
        K = np.array([0,0,1])
        N = np.cross(K,h)
        print(N)
        if N[1] > 0: # if Ny > 0 then RA is the cosine
            LAN = np.arccos(N[0]/norm(N))
        else:        # if Ny < 0 then RA is 2*pi - cosine
            LAN = 2*np.pi - np.arccos(N[0]/norm(N))

        # argument of perigee
        if e[2] > 0: # if ez > 0 then omega < 180
            omega = np.arccos( np.dot(e,N)/(e_mag*norm(N)) )
        else:        # if ez < 0 then omega > 180
            omega = 2*np.pi - np.arccos(np.dot(e,N)/(e_mag*norm(N)))

        # true anomaly
        v_radial = np.dot(v,r)/norm(r)  # radial velocity
        if v_radial > 0:
            theta = np.arccos(np.dot(e,r)/(e_mag*norm(r)))
        else:
            theta = 2*np.pi - np.arccos(np.dot(e,r)/(e_mag*norm(r)))


        # calculate ecc anomaly

        eccAnom = 2*np.arctan(np.sqrt((1-e_mag)/(1+e_mag))*np.tan(theta/2))

        meanAnom = eccAnom - e_mag*np.sin(eccAnom)


        elements = {'epoch':epoch,'h': h, 'i': i, 'e_vec': e, 'e': norm(e), 'LAN': LAN, 'omega': omega, 'theta': theta,'eccAnom':eccAnom, 'meanAnom': meanAnom}

        return elements


    
