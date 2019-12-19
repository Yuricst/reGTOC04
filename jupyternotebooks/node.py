
from orbit import Orbit, AU2km, mu, day2s, g0, ISP
import elements as el
import angles as an
from util import load_dataset, select_asteroid
from util import asteroids, earth, asteroid_ind

import lambert_combine as lb
#from lambert_izzo import lambert as lambert_izzo
#from lambert import lambert as lambert_bate

import sys

import numpy as np
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import logging
from scipy.optimize import minimize, LinearConstraint

# defined a helper function get the specific asteroid im looking for



class Node():
    """Node class defining sequences"""

    def __init__(self, epoch, index, approach_orbit=None, parent=None):
        self.epoch = epoch
        self.index = index
        self.asteroid = select_asteroid(index)

        self.approach_orbit = approach_orbit
        #this is the orbit that defined at parents epoch and position,
        # and gets me to this asteroid's position at required epoch

        self.parent_node = parent

        if self.parent_node is None:
            self.visited_list = [self.index]
            self.maneuvers=[]
            self.cost=0.0
        else:
            self.visited_list = self.parent_node.visited_list + [self.index]


    def get_children(self, rtol=0.1*AU2km, epoch_horizon=100):
        """Gets list of children nodes, which are the set of asteroids which may be potentially be visited next.
        1)
        The orbit that brought the spacecraft to the current asteroid & the permitted asteroids (not yet visited)
        are propagated for 100 days.
        2)
        The relative distance (dr) between the spacecraft and the permitted asteroids (asteroids not yet visited)
        are computed.
        3)
        At closest approach between the spacecraft and an asteroid, if dr is below a threshold distance (0.1AU),
        this asteroid is a 'child' of the current node.
        """
        permitted_asteroids = asteroid_ind.difference(set(self.visited_list)) # list of asteroids that are not visited yet

        # epoch list to propagate and search closest approach (min dr)
        epoch_lst = np.arange(self.epoch,self.epoch+epoch_horizon,10)

        # search for all permitted asteroids
        for astindx in permitted_asteroids:
            # create orbit object for asteroid
            ast_orbit = select_asteroid(astindx)

            # initialize relative distance list
            dr = [0,]*len(epoch_lst)

            # iterate over allowable wait-time until next asteroid
            for i, epoch in enumerate(epoch_lst):
                # state-vector of spacecraft
                r_sc, v_sc = self.approach_orbit.rv(epoch)
                # state-vector of asteroid
                r_ast, v_ast = ast_orbit.rv(epoch)
                # relative distance between spacecraft and asteroid
                dr[i] = norm(r_ast - r_sc)

            # find epoch at which closest approach for the asteroid occurs
            drmin, idxmin = min((drmin, idx) for (idx, drmin) in enumerate(dr))

            #print(f'ast: {astindx}, dr_min: {drmin/AU2km:2.3f}, epoch: {epoch_lst[idxmin]}')


            # check if closest approach is below threshold
            if drmin < rtol:

                # create next asteroid in the current visit sequence
                try:
                    child = self.create_next_node(astindx,epoch_lst[idxmin])
                    print(child)
                    yield child
                except:
                    print(f'Failed to generate child at astindex:{astindx}, epoch:{epoch_lst[idxmin]}')
                    pass



    def len_of_chain(self):
        """Function updates number of asteroids visited in current sequence"""
        if self.parent_node is None:
            return 0

        return 1 + self.parent_node.len_of_chain()


    def mass_at_node(self,m_in,dv,m0=1500,mprop=1000,Isp=ISP):
        """Function to compute remaining spacecraft mass with Tsiolkovski
        Args:
            m_in (float): mass of spacecraft before burn
            dv (float): required delta V [km/s]
            m0 (float): total (initial) wet mass [kg]
            mprop (float): propellant mass [kg]
            Isp (float): specific impulse [s]
        Return:
            (float): spacecraft mass in [kg] at current node
        """
        ve = Isp*g0/1000 # exhaust velocity [km/s]
        m_f = m_in * np.exp(-dv/ve)

        # if remaining mass at node has to be less than empty mass of spacecraft
        if m_node < (m0-mprop):
            raise Out_of_fuel

        return m_f


    def create_next_node(self, target_ind, target_epoch):
        """Function creates next node to be visited by solving Lambert, updating current sc mass
        Args:
            target_ind (int): next asteroid to be visited
            target_epoch (float): epoch at which next asteroid is to be visited [MJD]
        Returns:
            (node): node object of input asteroid to be visited at input epoch
        """
        t0 = self.epoch
        r0, v0 = self.approach_orbit.rv()

        # find next ast and get its position (vel not needed)
        ast = select_asteroid(target_ind)
        r_target, v_target = ast.rv(target_epoch)

        # solve lamberts problem
        v1, v2 = lb.lambert(mu, r0, r_target, (target_epoch-t0)*day2s)

        # note that a maneouver is needed at the initial epoch to set us on a rendevousz path with target ast
        dv = v1 - v0 # only the initial change in velocity needed

        #incrememental cost (delta V, mass)
        if self.parent_node is None:
            # starting at earth
            inc_cost = max(0, norm(dv)-4.0)
            # call mass_at_node to compute remaining mass after maneuver
            #m_node = mass_at_node(1500, inc_cost)   # FIXME - SAYS THAT mass_at_node() is not defined...
        else:
            inc_cost = norm(dv)
            # call mass_at_node?

        # create new orbit, index=-2 since it is a transfer
        o_new = Orbit(name=f'Trx{self.len_of_chain()+1:2.0f}')
        o_new.from_rv(target_epoch, r_target, v2)

        # create new node
        new_Node = Node(epoch = target_epoch, index = target_ind, approach_orbit=o_new, parent=self)

        # update cost and manouever history
        new_Node.maneuvers = self.maneuvers + [(t0, dv)]
        new_Node.cost = self.cost + inc_cost

        return new_Node


    def get_priority(self, dv=True, mf=False, count=False):
        """Function to prioritize based on branch cost"""
        # overall objective function is to maximise (number of asteroids + mf/m0)

        priority = 0

        if count: priority += self.len_of_chain()
        if mf   : priority += np.exp(-self.cost/(ISP*g0/1000))
        if dv   : priority += -self.cost

        return -priority


    def local_opt(self, opt_window=50,method=None,**kwargs):

        # first collect the ind history and epoch history

        ind_hist = self.get_ind_hist()
        epoch_hist = self.get_epoch_hist()

        # define the function to optimize
        # TODO: remove the linear constraint, use the time of flight method
        def f_path_cost(epochs, inds):

            node = self.create_node_from_list(t_list=epochs, ind_list=inds)

            return node.cost

        def create_constraint(epochs):
            #defines the matrix, and then returns the linear constraint
            A1=np.diagflat(-np.ones(len(epochs)), k=0)
            A2=np.diagflat(+np.ones(len(epochs)-1), k=1)
            A=(A1+A2)[:-1,:]
            return LinearConstraint(A, 0, np.inf, keep_feasible=True)

        lin_con = create_constraint(epoch_hist)

        bounds = [(t-opt_window, t+opt_window) for t in epoch_hist]

        try:
            sol=minimize(f_path_cost,epoch_hist, args=(ind_hist), method=method, bounds=bounds, constraints=(lin_con))
        except Exception as e:
            raise RuntimeError(f'Failed to local optimize, {e}')


        if sol.success:
            # generate new node
            node = self.create_node_from_list(t_list=sol.x, ind_list=ind_hist)

            return node

        raise RuntimeError(f'Local opt failed to find a solution {sol}')

        # function returns the final node, given a list of times and asteroids to hit

    def create_node_from_list(self, t_list, ind_list):
        """Returns a node, using the t_list and ind_list to specify the path.

        Args:
            t_list (list): Epochs (MJD) of depart/approach.
            ind_list (list): Indices of the asteroids to approach or depart from

        Returns:
            Node: Node that contains the full history of the path.

        """

        # create the initial orbit:
        initial = Orbit(name='Initial', index=ind_list[0])
        rv = select_asteroid(ind_list[0]).rv(t_list[0])
        initial.from_rv(t_list[0], *rv)

        # create the initial node:
        node = Node(t_list[0], index=ind_list[0], approach_orbit=initial, parent=None)

        # create all the legs:
        for t, ind in zip(t_list[1:],ind_list[1:]):
            node = node.create_next_node(target_ind = ind, target_epoch=t)

        return node

    def get_ind_hist(self):
        """Return a list of the asteroid indicies in the path.
        Implemented recursively

        Returns:
            list: list of asteroids visited (in order).
        """

        if self.parent_node is None:
            return [self.index]

        return  self.parent_node.get_ind_hist()+ [self.index]

    def get_epoch_hist(self):
        """Return a list of the epochs for visiting each asteroid in path.
        Implemented recursively

        Returns:
            list: list of visit/depart/arrive epochs for each asteroid
        """

        if self.parent_node is None:
            return [self.epoch]

        return  self.parent_node.get_epoch_hist()+ [self.epoch]

    # printing tools
    def __repr__(self):

        return f'MJD {self.epoch:6.2f}, {self.asteroid.index:6.0f},    {self.asteroid},'

    def history(self):

        # call print(node.history()) to get the reverse sequence of nodes
        if self.parent_node is None:
            return 'Epoch,         Index,    Name,\n\n' + repr(self)

        return f'{self.parent_node.history()}\n{self}'

    def plot(self, only_traj=False):

        if self.parent_node is None:
            plt.plot(0,0,'xy')
            plt.grid()
            ax=plt.gca()
            ax.set_aspect(1)

        else:
            self.parent_node.plot(only_traj)

        if not only_traj:
            #plot orbit of current asteroid,
            self.asteroid.plot(start=self.epoch)

        (rx, ry, rz), v = self.asteroid.rv(self.epoch)
        plt.text(rx/AU2km,ry/AU2km, f'{self.len_of_chain()}')

        #plot current approach trajectory
        if self.parent_node is not None:
            self.approach_orbit.plot(start=self.parent_node.epoch, end=self.epoch)
