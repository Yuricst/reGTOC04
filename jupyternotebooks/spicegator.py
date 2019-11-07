#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function for orbit propagation using SPICE function
"""
import numpy as np
import spiceypy as spice
import pandas as pd

# conversion between MJD and JD
def _mjd2jd(mjd):
    """function converts MJD (Modified Julian Date) to JD (Julian Date)
    Args:
        mjd (float or lst): float or list of floats of ephemerides in MJD to be converted to JD
    Returns:
        (float or float): float or list of ephemeris equivalent in JD
    """
    if type(mjd) == list: 
        jd = [el + 2400000.5 for el in mjd]
    else:
        jd = mjd + 2400000.5
    return jd


def _jd2mjd(jd):
    """function converts JD (Julian Date) to MJD (Modified Julian Date)
    Args:
        mjd (float or lst): float or list of floats of ephemerides in JD to be converted to MJD
    Returns:
        (float or float): float or list of ephemeris equivalent in MJD
    """
    if type(mjd) == list: 
        mjd = [el - 2400000.5 for el in jd]
    else:
        mjd = jd - 2400000.5
    return mjd


# propagator ... could also make a separate function just to compute dr? or an option?
def propagate_spice(etr_mjd, eldf, MU=1.32712440018*10**11, step=1000):
    """
    SPICE-powered orbit propagator developed for GTOC4
    Args:
        et_MJD (lst): list including start and end time of propagation (in MJD)
        step (float): steps of propagation
        eldf (pandas df): pandas dataframe of orbital elements to be propagated (expect spacecraft to be first row)
    Returns:
        (tuple): state-vector, dr
    """
    
    # 1 astronomical unit [AU] to [km]
    au2km = 1.49597870691*10**8    

    # convert time range from MJD to JD
    etr_jd = _mjd2jd((etr_mjd))
    # create time array
    etrsteps = [x * (etr_jd[1] - etr_jd[0])/step + etr_jd[0] for x in range(step)]
    # store number of bodies to propagate
    [bdy,tmp] = eldf.shape
    # initialize 3d numpy array to store state-vectors
        # first:  number of arrays = number of bodies to propagate
        # second: rows = timesteps
        # third:  columns = x, y, z, vx, vy, vz
    sv = np.zeros((bdy, step, 6))
    
    # propagate over time array
    for i in range(step):
        # prepare orbital elements for spice.conics() function
        for j in range(bdy):
            # convert orbital elements to spice-format
            rp = eldf.at[j,'a']*au2km * (1 - eldf.at[j,'e'])
            elts = np.array([rp, eldf.at[j,'e'], np.rad2deg(eldf.at[j,'i']), np.rad2deg(eldf.at[j,'LAN']), np.rad2deg(eldf.at[j,'omega']), np.rad2deg(eldf.at[j,'M0']), _mjd2jd(eldf.at[j,'Epoch']), MU])
            tmp = spice.conics(elts, etrsteps[i])
            
            # FIXME - store state-vector of one object into sv 3d numpy array
            for k in range(6):
                sv[(j,i,k)] = tmp[k]
        
        #state = spice.conics(elts, etrsteps[i])
    
    return sv
    
