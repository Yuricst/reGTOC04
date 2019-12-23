"""Combined Lambert solver
Prioritizes Dario Izzo's formulation.
If failed to converge, switch to Bate's formulation.
If one of the method is to be used, directly call:
	lambert_izzp(*args) or
	lambert_bate(*args) where
	*args are: mu, r1, r2, tof
"""


import numpy as np
from numpy.linalg import norm
from numpy import cross
import logging

import scipy.optimize as opt
from scipy.special import hyp2f1

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


# -------------------- PRIMARY FUNCTION -------------------- #
# tool to use izzo and Bate lambert algorithm
def lambert(*args):
	"""Function uses Izzo's formulation to solve Lambert problem. If it doesn't converge, function switches to Bate's formulation.
	"""
	try:
		return lambert_izzo(*args)
	except:
		try:
			# insert counter for number of times izzo algorithm has been diverted?
			return lambert_bate(*args)
		except:
			raise RunTimeError('Failed to solve Lambert\'s problem')


# -------------------- BATE'S FORMULATION -------------------- #
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

		return S
	elif z == 0:
		S = 1/6

		return S
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


# -------------------- IZZO'S FORMULATION -------------------- #
def lambert_izzo(mu, r1, r2, tof, M=0, numiter=35, rtol=1e-8, return_="short"):

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
