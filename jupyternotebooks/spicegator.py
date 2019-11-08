#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function for orbit propagation using SPICE function
"""
import numpy as np
import spiceypy as spice
import pandas as pd
import time

# conversion from km to au
au2km = 1.49597870691*10**8


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



def propagate_spice(etr_mjd, eldf, MU=1.32712440018*10**11, step=1000, sv_option=True, dr_option=True):
    """
    SPICE-powered orbit propagator developed for GTOC4
    Args:
        et_MJD (lst): list including start and end time of propagation (in MJD)
        step (float): steps of propagation
        eldf (pandas df): pandas dataframe of orbital elements to be propagated (expect spacecraft to be first row)
        sv_option (bool): if set to True, compute state-vector
        dr_option (bool): if set to True, compute relative position vector between object of first row and object of other rows
    Returns:
        (tuple): time-array [JD], state-vector (if True), relative position vector (if True), and relative position scalar
            state-vector is 3d numpy array, where:
                1st index: number of arrays = number of bodies to propagate
                2nd index: rows = timesteps
                3rd index: columns = x, y, z, vx, vy, vz, name of object
            relative position vector is also 3d numpy array, where:
                1st index: number of arrays = number of bodies relative to spacecraft
                2nd index: rows = timesteps
                3rd index: columns = dx, dy, dz, name of object
            and relative position salar is 3d numpy array, where: 
                1st index: number of arrays = number of bodies relative to spacecraft
                2nd index: rows = timesteps
                3rd index: name of object
    Examples:
        et, sv, dr, drnorm = propagate_spice(etr_MJD, el_pd1, MU=1.32712440018*10**11, step=steps, sv_option=True, dr_option=True)
    """
    
    # measure time at start of program
    tstart = time.time()
    
    # convert time range from MJD to JD
    etr_jd = _mjd2jd((etr_mjd))
    # create time array
    etrsteps = [x * (etr_jd[1] - etr_jd[0])/step + etr_jd[0] for x in range(step)]
    # store number of bodies to propagate
    [bdy,tmp] = eldf.shape
    
    # initialize 3d numpy array to store state-vectors and object name
    if sv_option == True:
        sv = np.zeros((bdy, step, 7))
        
    # initialise 3d numpy array to store relative position vector and object name
    if dr_option == True:
        dr = np.zeros((bdy-1, step, 4))
        drnorm = np.zeros((bdy-1, step, 2))
    
    # propagate over time array
    for i in range(step):
        # prepare orbital elements for spice.conics() function
        for j in range(bdy):
            # convert orbital elements to spice-format
            rp = eldf.at[j,'a']*au2km * (1 - eldf.at[j,'e'])
            elts = np.array([rp, eldf.at[j,'e'], np.rad2deg(eldf.at[j,'i']), np.rad2deg(eldf.at[j,'LAN']), np.rad2deg(eldf.at[j,'omega']), np.rad2deg(eldf.at[j,'M0']), _mjd2jd(eldf.at[j,'Epoch']), MU])
            tmp = spice.conics(elts, etrsteps[i])
            
            # FIXME - store state-vector of one object
            if sv_option == True:
                for k in range(6):
                    sv[(j,i,k)] = tmp[k]
                sv[j,i,6] = eldf.at[j,'Name']
                    
            # store relative state-vector of current object (except if object is the spacecraft ifself)
            if dr_option == True:
                if j == 0:
                    sc_currentpos = np.zeros(3)
                    # store current spacecraft location
                    sc_currentpos[0] = tmp[0]  # state-vector[0]
                    sc_currentpos[1] = tmp[1]  # state-vector[1]
                    sc_currentpos[2] = tmp[2]  # state-vector[2]
                else:
                    # compute relative vector
                    for l in range(3):
                        dr[(j-1,i,l)] = tmp[l] - sc_currentpos[l]
                    dr[(j-1,i,3)] = eldf.at[j,'Name']
                    drnorm[(j-1,i,0)] = np.sqrt( dr[(j-1,i,0)]**2 + dr[(j-1,i,1)]**2 + dr[(j-1,i,2)]**2 )
                    drnorm[(j-1,i,1)] = eldf.at[j,'Name']
                    
    # measure time
    tend = time.time()
    # computation time
    dt = tend - tstart
    # print computational time
    print(f'Propagation time: {round(dt,2)} [sec]')
    
    if sv_option == True and dr_option == True:
        return etrsteps, sv, dr, drnorm
    elif sv_option == True and dr_option == False:
        return etrsteps, sv
    elif sv_option == False and dr_option == True:
        return etrsteps, dr, drnorm
    
