import functools
import numpy as np
from numpy import pi, sqrt, cross
from numpy.linalg import norm
import matplotlib.pyplot as plt

import elements as el
from angles import TA2MA, MA2TA

from memo import memoized


mu = 1.32712440018e11
day2s = 86400
AU2km = 1.49597870691e8
ISP = 3000*9.80665


# memoizer to store outputs as they are solved for
# note that this decorator ignores **kwargs
# taken from https://wiki.python.org/moin/PythonDecoratorLibrary#Memoize

def _array2string(x, format):
    #format is in the style of "%.8f"
    return np.array2string(x, formatter={'float_kind':lambda x: format % x})

class Orbit():

    def __init__(self, name=None, mu=mu, index=-2):

        self.index = index
        self.name = name
        self.mu = mu

        return

    # the orbit is define at some point of time using some set_orb_elements
    # stored such that orbit is defined at any given epoch0
    # constant orbital elements are just defined as is
    # varying ones are defined with 0 to indicate they are evaluated at epoch 0

    def from_coe(self, epoch, h, e, LAN, inc, argp, trueAnom):

        self.epoch0 = epoch
        self.h = h
        self.e = e
        self.LAN = LAN
        self.inc = inc
        self.argp = argp
        self.trueAnom0 = trueAnom

        # set initial mean anomaly
        self.meanAnom0 = TA2MA(TA=self.trueAnom0, ecc=self.e)

        # set semi-major axis
        self.a = self.h**2/self.mu/(1 - self.e**2);

        # determine initial position and velocity
        self.r0, self.v0 = el.coe2rv(self.mu, h, e, LAN, inc, argp, trueAnom)

    def from_rv(self, epoch, r, v):

        self.epoch0 = epoch
        self.r0 = r
        self.v0 = v

        # convert to orbital elements
        self.h, self.e, self.LAN, self.inc, self.argp, self.trueAnom0 = el.rv2coe(self.mu, r, v)

        # set initial mean anomaly
        self.meanAnom0 = TA2MA(TA=self.trueAnom0, ecc=self.e)

        # set semi-major axis
        self.a = self.h**2/self.mu/(1 - self.e**2);

    def from_gtoc(self, name, epoch, a, e, inc, LAN, argp, meanAnom0):
        self.name = name
        self.epoch0 = epoch
        self.a = a*AU2km
        self.e = e
        self.h = sqrt(self.mu*self.a*(1-self.e**2))
        self.inc = inc * pi/180
        self.LAN = LAN * pi/180
        self.argp = argp * pi/180
        self.meanAnom0 = meanAnom0 * pi/180

        #convert meanAnom to trueAnom
        self.trueAnom0 = MA2TA(self.meanAnom0, self.e)
        # determine initial position and velocity
        self.r0, self.v0 = el.coe2rv(self.mu, self.h, self.e, self.LAN, self.inc, self.argp, self.trueAnom0)

    @memoized
    def propagate(self, epoch):
        """ Propagates orbit forward to some epoch"""

        ## TODO: Implement

        raise NotImplementedError


    @memoized
    def rv(self, epoch=None, tol=1e-2):
        """Propagate orbit forward in time with Kepler's equation"""
        # memoize: if all arguments to rv() is the same as sometime before, it will return previsouly computed value
        # >>> may be handy when times between subsequent nodes overlap

        # for ease
        if epoch is None:
            return self.r0, self.v0

        tof = (epoch - self.epoch0)*day2s

        if np.abs(self.e-1.0) > tol:
            # either elliptic or Hyperbolic
            a = self.a

            meanAnom = self.meanAnom0 + tof * sqrt(self.mu/np.abs(self.a**3))

            trueAnom = MA2TA(MA=meanAnom, ecc=self.e)

        else:
            # close to a parabolic trajectory
            p = np.dot(self.h,self.h) / self.mu  # semi-latus rectum, parameter of orbit
            q = p * np.abs(1.0 - self.e) / np.abs(1.0 - self.e ** 2)
            # mean motion n = sqrt(mu / 2 q^3) for parabolic orbit
            meanAnom = self.meanAnom0 + tof * np.sqrt(self.mu / 2.0 / (q ** 3))
            trueAnom = MA2TA(meanAnom, self.e)

        coe = [self.h, self.e, self.LAN, self.inc, self.argp, trueAnom]

        return el.coe2rv(self.mu, *coe) # returns the state-vector at new trueAnom (and new Epoch)


    def __repr__(self):

        return f'{self.name}'

    def details(self):

        s = f"Orbit: {self.name}"
        s += f"\n Epoch     :  {self.epoch0}"
        s += f"\n a         :  {self.a/AU2km:.8f} AU"
        s += f"\n e         :  {self.e:.8f}"
        s += f"\n inc       :  {self.inc*180/pi:.8f} deg"
        s += f"\n LAN       :  {self.LAN*180/pi:.8f} deg"
        s += f"\n argp      :  {self.argp*180/pi:.8f} deg"
        s += f"\n meanAnom0 :  {self.meanAnom0*180/pi:.8f} deg"
        s += f"\n trueAnom0 :  {self.trueAnom0*180/pi:.8f} deg"
        s += f"\n h         :  {self.h:.8f} km2/s"
        s += f"\n r0        :  {_array2string(self.r0/AU2km, format='%.8f')} AU"
        s += f"\n v0        :  {_array2string(self.v0, format='%.8f')} km/s"

        return s

    def plot(self, start=None, end=None, dim=2, num=100, **kwargs):
        """plots orbit in 2D between start and end time, with num points
        Args:
            start (float): start epoch of orbit to be plotted (MJD)
            end (float): end epoch of orbit to be plotted (MJD)
            dim (int): dimension of plot (either 2D or 2D)
            num (int): number of points to construct the orbit
        """

        if start is None:
            start = self.epoch0
        if end is None:
            if self.e > 0.99: #close to parabolic or hyperbolic
                period = 2*365 # 2 years enforced
            else:
                period = (2*pi*sqrt(self.a**3/self.mu))/day2s
            end = start + period

        # period is in days
        # find terminal time by projecting one period, and simulating

        ts = np.linspace(start, end, num)

        rx = [None, ]*len(ts)
        ry = [None, ]*len(ts)
        rz = [None, ]*len(ts)

        for i, t in enumerate(ts):
            r, v = self.rv(t)
            r = r/AU2km
            rx[i], ry[i], rz[i] = r

        # plot in 2D
        if dim == 2:
            p = plt.plot(rx, ry)[0]

            plt.plot(rx[0],ry[0],'-o',color=p.get_color(), label=f'{start:5.2f}MJD: {self.name}', **kwargs)
            plt.xlabel('x [AU]')
            plt.ylabel('y [AU]')

        # plot in 3D - FXIME!
        if dim == 3:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(rx[0],ry[0],rz[0],'-o', label=f'{self.name} at {start}MJD')
