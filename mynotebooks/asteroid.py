import numpy as np
import scipy as sp
import pandas as pd
import scipy.optimize as opt


class Asteroid:
    def __init__(self, name, epoch, a_AU, e, i_deg, LAN_deg, argPeri_deg, meanAnom_deg):
        '''Class: Define an asteriod (or earth) using elements.
        Format agrees with the columns of GTOC4 dataset'''

        self.mu = 1.32712440018e+11  # km3/s2
        self.u_AU = 1.29597870691e+8  # km
        self.u_day = 86400  # seconds
        self.u_deg2rad = np.pi/180

        self.name = name
        self.epoch = epoch  # in MJD
        self.a = a_AU*self.u_AU
        self.e = e
        self.i = self.u_deg2rad*i_deg
        self.LAN = self.u_deg2rad*LAN_deg
        self.omega = self.u_deg2rad*argPeri_deg
        self.M0 = self.u_deg2rad*meanAnom_deg

        self.elements = (self.epoch, self.a, self.e, self.i, self.LAN, self.omega, self.M0)

        # cache the last calculated results
        self.last_epoch_meanAnom = None
        self.last_epoch_eccAnom = None
        self.last_epoch_trueAnom = None
        self.last_epoch_r_mag = None
        self.last_epoch_gamma = None
        self.last_epoch_v_mag = None
        self.last_epoch_r = None
        self.last_epoch_v = None

    def get_meanAnom(self, epoch):
        '''function returns mean anomaly
        Args:
            epoch (float): time, MJD
        Returns:
            (float): mean anomaly, radians (wrapped 0 .. 2 pi)
        '''

        if epoch == self.last_epoch_meanAnom:
            return self.last_meanAnom
        self.last_epoch_meanAnom = epoch

        meanAnom = self.M0 + np.sqrt(self.mu/self.a**3)*((epoch-self.epoch)*self.u_day)

        meanAnom = meanAnom % (2*np.pi)

        self.last_meanAnom = meanAnom

        return meanAnom

    def get_eccAnom(self, epoch, method=None, **kwargs):
        ''' Function calculates the eccentric anomaly. Uses scipy.optimize.root_scalar internally.
        Args:
            epoch (float): time, MJD
            method (str): method for root solving. optional, defaults to None.
        Returns:
            (float): mean anomaly, radians (wrapped 0 .. 2 pi), if solution converged
        Errors:
            if convergence or solver fails, an error is raised.
        '''

        if epoch == self.last_epoch_eccAnom:
            return self.last_eccAnom

        self.last_epoch_eccAnom = epoch

        def root_eccAnom(eccAnom, meanAnom, ecc):
            '''Returns the error in Kepler's equation, and its first and second derivatives'''

            error = eccAnom - ecc*np.sin(eccAnom) - meanAnom
            deriv = 1 - ecc*np.cos(eccAnom)
            deriv2 = ecc*np.sin(eccAnom)
            return error, deriv, deriv2

        # calculate mean anom
        meanAnom = self.get_meanAnom(epoch)

        # solve for true anom
        sol = opt.root_scalar(root_eccAnom, args=(meanAnom, self.e), bracket=(
            0, 2*np.pi), fprime2=True, method=method, **kwargs)

        if sol.converged:

            self.last_eccAnom = sol.root % (2*np.pi)

            return sol.root % (2*np.pi)
        else:
            print(f'Failed at: {str(self)} at epoch: {epoch}')
            print(sol)
            self.last_eccAnom = sol

            raise RuntimeError('Eccentric anomaly calculation failed:')

    def get_trueAnom(self, epoch, **kwargs):
        '''Returns true anomaly at epoch.
        Args:
            epoch (float): time, MJD
        Returns:
            (float): true anomaly, radians (wrapped 0 .. 2 pi)
        '''

        if epoch == self.last_epoch_trueAnom:
            return self.last_trueAnom
        self.last_epoch_trueAnom = epoch

        eccAnom = self.get_eccAnom(epoch, **kwargs)

        e = self.e

        trueAnom = (np.arctan((1/np.sqrt((1-e)/(1+e))) * np.tan(eccAnom/2))) % 2*np.pi

        self.last_trueAnom = trueAnom

        return trueAnom

    def get_r_mag(self, epoch=None, trueAnom=None, **kwargs):
        '''Returns the distance from the sun.
        If only epoch is provided, trueAnom is calculated.
        If trueAnom is provided, epoch has no effect
        Args:
            epoch (float): time, MJD
            trueAnom (float): true anomaly, rad

        Returns:
            (float): distance from sun, km
        '''

        if epoch == self.last_epoch_r_mag:
            return self.last_r_mag
        self.last_epoch_r_mag = epoch

        if trueAnom is None:
            trueAnom = self.get_trueAnom(epoch, **kwargs)

        r_mag = self.a*(1-self.e**2)/(1+self.e*np.cos(trueAnom))

        self.last_r_mag = r_mag

        return r_mag

    def get_gamma(self, epoch, trueAnom=None, **kwargs):
        '''Returns flight path angle gamma.
        If only epoch is provided, trueAnom is calculated.
        If trueAnom is provided, epoch has no effect
        Args:
            epoch (float): time, MJD
            trueAnom (float): true anomaly, rad

        Returns:
            (float): flight path angle, rad
        '''

        if epoch == self.last_epoch_gamma:
            return self.last_gamma
        self.last_epoch_gamma = epoch

        if trueAnom is None:
            trueAnom = self.get_trueAnom(epoch, **kwargs)

        gamma = np.arctan((self.e*np.sin(trueAnom))/(1+self.e*np.cos(trueAnom)))

        self.last_gamma = gamma

        return gamma

    def get_v_mag(self, epoch, trueAnom=None, **kwargs):
        '''Returns helio-centric speed of asteroid.
        If only epoch is provided, trueAnom is calculated.
        If trueAnom is provided, epoch has no effect
        Args:
            epoch (float): time, MJD
            trueAnom (float): true anomaly, rad

        Returns:
            (float): speed of asteroid in orbit about Sun, km/s
        '''

        if epoch == self.last_epoch_v_mag:
            return self.last_v_mag
        self.last_epoch_v_mag = epoch

        r = self.get_r_mag(epoch, trueAnom, **kwargs)

        v = np.sqrt(2*self.mu/r - self.mu/self.a)

        self.last_v_mag = v

        return v

    def get_r(self, epoch, trueAnom=None, **kwargs):
        '''Returns position vector relative to Sun.
        If only epoch is provided, trueAnom is calculated.
        If trueAnom is provided, epoch has no effect
        Args:
            epoch (float): time, MJD
            trueAnom (float): true anomaly, rad

        Returns:
            (numpy.array): position vector, km
        '''

        if epoch == self.last_epoch_r:
            return self.last_r
        self.last_epoch_r = epoch

        if trueAnom is None:
            trueAnom = self.get_trueAnom(epoch, **kwargs)

        r = self.get_r_mag(epoch, trueAnom, **kwargs)

        costo = np.cos(trueAnom + self.omega)
        sinto = np.sin(trueAnom + self.omega)

        x = r*(costo*np.cos(self.LAN) - sinto*np.cos(self.i)*np.sin(self.LAN))
        y = r*(costo*np.sin(self.LAN) + sinto*np.cos(self.i)*np.cos(self.LAN))
        z = r*(sinto*np.sin(self.i))

        r_vec = np.array([x, y, z])

        self.last_r = r_vec

        return r_vec

    def get_v(self, epoch, trueAnom=None, **kwargs):
        '''Returns heliocentric velocity vector.
        If only epoch is provided, trueAnom is calculated.
        If trueAnom is provided, epoch has no effect
        Args:
            epoch (float): time, MJD
            trueAnom (float): true anomaly, rad

        Returns:
            (numpy.array): heliocentric velocity vector, km/s
        '''

        if epoch == self.last_epoch_v:
            return self.last_v

        self.last_epoch_v = epoch

        if trueAnom is None:
            trueAnom = self.get_trueAnom(epoch, **kwargs)

        v = sef.get_v_mag(epoch, trueAnom)

        gamma = self.get_gamma(epoch, trueAnom)

        costog = np.cos(trueAnom + self.omega - gamma)
        sintog = np.sin(trueAnom + self.omega - gamma)

        vx = v*(-sintog*np.cos(self.LAN) - costog*np.cos(self.i)*np.sin(self.LAN))
        vy = v*(-sintog*np.sin(self.LAN) + costog*np.cos(self.i)*np.cos(self.LAN))
        vz = v*(costog*np.sin(self.i))

        v_vec = np.array([vx, vy, vz])

        self.last_v = v_vec

        return v_vec

    def dist_to(self, other, epoch, **kwargs):
        '''Returns distance vector to the other asteroid.
        Uses the other objects .get_r(epoch) function to get its position

        Args:
            other (Asteroid): other asteroid object
            epoch (float): time, MJD

        Returns:
            (numpy.array): (r_other-r_self) vector, km
        '''

        r = self.get_r(epoch, **kwargs)
        rp = other.get_r(epoch, **kwargs)

        return rp-r

    def dist_to_mag(self, other, epoch, **kwargs):
        '''Returns the magnitude of the distance to the other asteroid
        Uses the other objects .get_r(epoch) function to get its position

        Args:
            other (Asteroid): other asteroid object
            epoch (float): time, MJD
        Returns:
            (float): distance between objects, km
        '''

        dr = self.dist_to(other, epoch, **kwargs)

        return np.linalg.norm(dr)

    def details(self):
        '''String output of orbital elements of the asteroid
        Returns:
            (str): asteroid details
        '''

        out = f'*Asteroid {self.name}*'
        out += f'\n  Epoch0 (MJD)  : {self.epoch}'
        out += f'\n  a (AU)        : {self.a/self.u_AU}'
        out += f'\n  e (deg)       : {self.e/self.u_deg2rad}'
        out += f'\n  i (deg)       : {self.i/self.u_deg2rad}'
        out += f'\n  LAN (deg)     : {self.LAN/self.u_deg2rad}'
        out += f'\n  argPeri (deg) : {self.omega/self.u_deg2rad}'
        out += f'\n  meanAnom (deg): {self.M0/self.u_deg2rad}'

        return out

    def __repr__(self):

        out = f'Asteroid {self.name}'

        return out
