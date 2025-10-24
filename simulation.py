###############################################################################
#
# simulation.py
#
# @Function: This file contains all the code to generate a simulation with
# particles
#
# A simulation contains particles
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from numba import types, typed, cuda
from numba.experimental import jitclass
import os
import pandas

from constants import *
import quantum


class simulation:
    """
    Default class for simulation. Contains the default structor for a simulation.
    Draws a basic 1D particle simulation as
    example.  Should be inherited.
    """

    def __init__(self, del_x = 0.1e-9, dt = 8e-17, fdtd = quantum.fdtd, sim_size = None, sim_length=None):
        self.name = "default_sim"
        self.cache_dir = f"/tmp/quantum_sim/data/{self.name}/"
        self.output_dir = "images"
        self.particles = []
        self.v_fields = []
        self.v_field_total = np.zeros(sim_length)
        self.ra = (0.5*hbar_J/m0)*(dt/del_x**2)
        self.rd = dt/hbar_J
        self.abs = None
        self.abs_func = None
        self.dt = dt
        self.del_x = del_x
        if sim_size != None:
            self.sim_size = sim_size
        elif sim_size != None:
            self.sim_size = int(sim_length/del_x)
        self.sim_size_spat = self.sim_size*del_x

        self.n_steps = 0
        self.sim_time = 0
        self.KE_total = 0
        self.PE_total = 0
        self.log = []
        self.logging = False
        self.log = []

        # dft
        self._dft = False
        self.dft_points = []
        self.dfts = []
        self.dft_E = []

        self.sim_space = np.linspace(self.del_x, self.sim_size_spat, self.sim_size)


    def run(self, time=None, steps=None, save_each_step=False):
        # run simulation for set time in ps

        states = []
        if steps == None:
            n_step = time/self.dt
        elif time == None:
            n_step = steps
        else:
            n_step = 1

        for step in range(n_step):
            # V field calulation
            self.v_field_total = np.zeros(self.sim_size)
            for field in self.v_fields:
                field.step(self.sim_time)
                self.v_field_total += field.V_field

            # run time dependent initialization functions
            for particle in self.particles:
                if particle.time_init:
                    particle.time_initlize(self.sim_time)
            
            # run particles
            ind = 0
            for particle in self.particles:
                temp_particle_list = self.particles.copy()
                temp_particle_list.remove(particle)
                if temp_particle_list == []:
                    temp_particle_list.append(-1)
                particle.fdtd(self.v_field_total, temp_particle_list, self.ra, self.rd, abs = self.abs)
                measureables = particle.update_measurables(self.dt, self.del_x, self.v_field_total)
                # if dft
                if self._dft:
                    for i in range(len(self.dft_points)):
                        self.dfts[i] += particle.run_dft_at_point(self.dft_E[i],
                                                          self.dft_points[i], self.sim_time)

                # calulate extra things
                # if logging
                if self.logging:
                    self.log[ind].append((self.sim_time, self.n_steps, self.particles, self.v_field_total, *measureables))
                if save_each_step:
                    states.append((self.sim_time, self.n_steps, self.particles, self.v_field_total, *measureables))
        

            self.n_steps += 1
            self.sim_time += self.dt
        
        if save_each_step:
            return states
        
        return self.particles, self.v_field_total, (self.sim_time, self.n_steps)

    def initize_particle(self, particle, init_function):
        # initialize particles
        particle.sim_size = self.sim_size
        init_function.dt = self.dt
        init_function.del_x = self.del_x
        particle.initialize(init_function)
        self.particles.append(particle)
        self.log.append([]) 

    def output_to_csv(self, name):
        ind = 0
        for log in self.log:
            file = open(f"{ind}_{name}", "w")
            file.write("sim_time,step,ptot,ke,pe,H\n")
            for line in log:
                line = [*line[0:2], *line[5:len(line)]]
                line = list(map(lambda l : str(l), line))
                file.write(",".join(line)+ "\n")
            file.close()

    def initize_abs(self, abs):
        self.abs = abs.abs(self.sim_space)
        self.abs_func = abs

    def init_dft(self, start, samples, end=0, dt=0, loc=0, pos=0):
        if loc == 0:
            loc = int(pos/self.del_x)
        self._dft = True
        self.dft_points.append(loc)

        if dt == 0:
            dt = (end - start)/samples
            self.dft_E.append(np.arange(start, end+dt, dt))
        elif end == 0:
            E = np.zeros(samples)
            for d in range(samples):
                E[d] = dt*(d+1)
            self.dft_E.append(E)
        self.dfts.append(np.zeros((len(self.dft_E[-1])), dtype=complex))

    def init_particle(self, particle, init_function):
        # initialize particles
        particle.sim_size = self.sim_size
        init_function.dt = self.dt
        init_function.del_x = self.del_x
        particle.initialize(init_function)
        self.particles.append(particle)
        self.log.append([]) 

    def init_vfield(self, v_field):
        v_field.sim_size = self.sim_size
        v_field.del_x = self.del_x
        v_field.init_eqt()
        self.v_fields.append(v_field)

    def draw_sim(self):
        # allow user to make own drawing code
        pass

    def create_video(self):
        pass

    def _save(self):
        if not os.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def _wipe(self):
        if os.isdir(self.cache_dir):
            os.rmdir(self.cache_dir)
    
    def sumerize(self):
        # output the results and varables of the simulation in a csv compatable format
        pass


class Vfield:
    """
    Base class for potential fields in the simulation
    """
    def __init__(self, eqt):
        self.V_field = eqt
    
    def step(self, time):
        pass


class aabs:
    """
    adds the barrier at the start and end of the sim to remove reflections
    """
    sim_space = None
    def __init__(self, eqt_s, start, end_offset=0, eqt_e=None):
        self.start_loc = start
        if end_offset != 0:
            self.end_loc = end_offset
        else:
            self.end_loc = start
        
        self.start_eqt = eqt_s
        if eqt_e == None:
            self.end_eqt = eqt_s
        else:
            self.end_eqt = eqt_e

    def abs(self, sim_space):
        ret_abs = np.ones(len(sim_space))
        for n in range(int(len(sim_space)/2)):
            if n < self.start_loc:
                ret_abs[n] = self.start_eqt(n)
            if (len(sim_space)-1) - n > len(sim_space) - self.end_loc:
                ret_abs[(len(sim_space)-1) - n] = self.end_eqt(n)
        return ret_abs

        

class xabs(aabs):
    """
    abs normally used
    """

    def __init__(self, start, end_offset=0):
        self.start = start
        self.start_eqt = lambda n :1 - .5*((self.start_loc - n)/self.start_loc)**3
        super().__init__(self.start_eqt, start)


