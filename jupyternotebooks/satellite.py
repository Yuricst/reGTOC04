import numpy as np
import scipy as sp
import pandas as pd
import scipy.optimize as opt
from numpy import linalg as linalg
from numpy.linalg import norm

from asteroid import Asteroid

class Satellite(Asteroid):

    def __init__(self):

        self.mu = 1.32712440018e+11  # gravitational parameter of Sun km3/s2
        self.u_AU = 1.4959780691e+8   # 1 AU in km
        self.u_day = 86400           # 1 day in seconds
        self.u_deg2rad = np.pi/180   # conversion from degree to radians

        self.r = None
        self.v = None

        self.name = 'Spacecraft Scaramouche'
        self.epoch = None           # must be in MJD
        self.a = None
        self.e_mag = None
        self.i = None
        self.LAN = None
        self.omega = None
        self.M0 = None

        #self.elements = (self.epoch, self.a, self.e_mag, self.i, self.LAN, self.omega, self.M0)

        # cache the last calculated results
        self.last_epoch_meanAnom = None
        self.last_epoch_eccAnom = None
        self.last_epoch_trueAnom = None
        self.last_epoch_r_mag = None
        self.last_epoch_gamma = None
        self.last_epoch_v_mag = None
        self.last_epoch_r = None
        self.last_epoch_v = None

    def get_state(self):

        return self.epoch, self.r, self.v

    def set_state(self, epoch, r, v, add_to_hist=False):

        self.epoch = epoch
        self.r  = r
        self.v  = v

        # also update all orb elements
        self.set_orb_elements(self.epoch, self.r, self.v)

    def set_orb_elements(self, epoch, r, v):

        #calculate angular momentum
        self.h = h = np.cross(r,v)

        #calculate inclination
        self.i = i = np.arccos(h[2]/norm(h))

        # eccentricity
        self.e = e = np.cross(v, h)/self.mu - r/norm(r)
        self.e_mag = e_mag = norm(e)

        # semi major axis
        self.a = a = (norm(h)**2/self.mu)/(1-e_mag**2)

        # right ascension
        K = np.array([0,0,1])
        N = np.cross(K,h)

        if N[1] > 0: # if Ny > 0 then RA is the cosine
            self.LAN = LAN = np.arccos(N[0]/norm(N))
        else:        # if Ny < 0 then RA is 2*pi - cosine
            self.LAN = LAN = 2*np.pi - np.arccos(N[0]/norm(N))

        self.LAN = self.LAN % (2*np.pi)

        # argument of perigee
        if e[2] > 0: # if ez > 0 then omega < 180
            self.omega = omega = np.arccos( np.dot(e,N)/(e_mag*norm(N)) )
        else:        # if ez < 0 then omega > 180
            self.omega = omega = 2*np.pi - np.arccos(np.dot(e,N)/(e_mag*norm(N)))

        self.omega = self.omega % (2*np.pi)

        # true anomaly
        v_radial = np.dot(v,r)/norm(r)  # radial velocity
        if v_radial > 0:
            theta = np.arccos(np.dot(e,r)/(e_mag*norm(r)))
        else:
            theta = 2*np.pi - np.arccos(np.dot(e,r)/(e_mag*norm(r)))

        theta =  theta % (2*np.pi)

        self.theta = theta

        # calculate ecc anomaly

        self.eccAnom = eccAnom = (2*np.arctan(np.sqrt((1-e_mag)/(1+e_mag))*np.tan(theta/2))) % (2*np.pi)

        # calculate mean anom
        self.M0 = (eccAnom - e_mag*np.sin(eccAnom)) % (2*np.pi)
