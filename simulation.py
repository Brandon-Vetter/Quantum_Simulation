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
#import pandas
import yaml

from constants import *
import particle
import v_field


class Simulation:
    """
    Default class for simulation. Contains the default structor for a simulation.
    Draws a basic 1D particle simulation as
    example.  Should be inherited.
    """
    sim_size = 0
    sim_size_spat = 0
    sim_mid = 0
    sim_mid_spat = 0
    sim_space = 0

    def __init__(self, del_x = 0.1e-9, dt = 8e-17, sim_size = None, sim_length=None):

        # simulation setup
        self.particles = []
        self.v_fields = []
        self.v_field_total = np.zeros(sim_length)
        self.v_field_total_eV = np.zeros(sim_length)
        self.ra = (0.5*hbar_J/m0)*(dt/del_x**2)
        self.rd = dt/hbar_J
        self.dt = dt
        self.del_x = del_x
        if sim_size != None:
            self.sim_size = sim_size
        elif sim_length != None:
            self.sim_size = int(sim_length/del_x)
        
        if sim_size != None or sim_length != None:

            self.sim_size_spat = self.sim_size*del_x
            self.sim_mid = int(self.sim_size/2)
            self.sim_mid_spat = self.sim_mid*del_x
            self.sim_space = np.linspace(self.del_x, self.sim_size_spat, self.sim_size)

        # abc setup
        self.abc = None
        self.abc_func = None

        # measurables
        self.n_steps = 0
        self.sim_time = 0
        self.KE_total = 0
        self.PE_total = 0

        # logging
        self.logging = False
        self.log = []

        # dft
        self._dft = False
        self.dft_points = []
        self.dfts = []
        self.dft_E = []

    def _calulate_total_vfields(self):
        # V field calulation
        self.v_field_total = np.zeros(self.sim_size)
        self.v_field_total_eV = np.zeros(self.sim_size)
        time_vfield = True
        for field in self.v_fields:
            err_value = field.step(self.sim_time)
            self.v_field_total += field.v_field
            self.v_field_total_eV += J2eV*np.array(field.v_field)
    
    def run(self, time=None, steps=None, save_each_step=False, show_progress=True, progress_update=10):
        # run simulation for set time in ps

        states = []
        if steps == None:
            n_step = int(time/self.dt)
        elif time == None:
            n_step = steps
        else:
            n_step = 1


        for step in range(n_step):
            self._calulate_total_vfields()

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
                
                particle.fdtd(self.v_field_total, temp_particle_list, self.ra, self.rd, abc = self.abc, steps=1)
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

            if show_progress and (step+1)%progress_update == 0:
                persent = ((step+1)/n_step)*100
                bar = "#"*int(persent) + " "*(100-int(persent))
                print(f"{persent : .2f}% [{bar}]", end="\r")
                if persent >= 100:
                    print(" "*200, end="\r\n")

            self.n_steps += 1
            self.sim_time += self.dt
        
        if save_each_step:
            return states
        
        return self.particles, self.v_field_total, (self.sim_time, self.n_steps)

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

    def init_abc(self, abc):
        abc.del_x = self.del_x
        self.abc = abc.abc(self.sim_space)
        self.abc_func = abc

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
        init_function.initialize()
        particle.initialize(init_function)
        self.particles.append(particle)
        self.log.append([]) 

    def init_vfield(self, v_field):
        v_field.sim_size = self.sim_size
        v_field.del_x = self.del_x
        v_field.init_eqt()
        self.v_fields.append(v_field)
    
    def save_sim(self, simname):
        if not os.path.isdir(simname):
            os.mkdir(simname)
        os.chdir(simname)

        sim_file = open(f"{simname}.sim", "w")
        ind = 0
        for par in self.particles:
            par.save_particle(f"{simname}_{ind}")
            ind += 1
        
        sim_data = {
            "vfields" : list(map(lambda V : V.v_field, self.v_fields)),
            "dt" : self.dt,
            "del_x" : self.del_x,
            "sim_size" : self.sim_size,
            "n_steps" : self.n_steps,
            "abc" : self.abc,
            "dfts" : self.dfts,
            "dft_E" : self.dft_E,
            "dft_points" : self.dft_points,
            "_dft" : self._dft
        }
        yaml.dump(sim_data, sim_file)
        sim_file.close()
        os.chdir("..")

    def load_sim(self, simname):
        if not os.path.isdir(simname):
            return None

        os.chdir(simname)
        sim_file = open(f"{simname}.sim", "r")



        sim_data = yaml.load(sim_file, Loader=yaml.UnsafeLoader)

        self.dt = sim_data["dt"]
        self.del_x = sim_data["del_x"]
        self.sim_size = sim_data["sim_size"]
        self.sim_size_spat = self.sim_size*self.del_x
        self.sim_mid = int(self.sim_size/2)
        self.sim_mid_spat = self.sim_mid*self.del_x
        self.n_steps = sim_data["n_steps"]
        self.abc = sim_data["abc"]
        self.dfts = sim_data["dfts"]
        self.dft_E = sim_data["dft_E"]
        self.dft_points = sim_data["dft_points"]
        self._dft = sim_data["_dft"]
        self.ra = (0.5*hbar_J/m0)*(self.dt/self.del_x**2)
        self.rd = self.dt/hbar_J
        self.v_fields = []
        for field in sim_data["vfields"]:
            vfield = v_field.Vfield()
            vfield.v_field = field
            self.v_fields.append(vfield)

        self.sim_space = np.linspace(self.del_x, self.sim_size_spat, self.sim_size)

        self._calulate_total_vfields()

        for particle_name in os.listdir("."):
            if ".part" not in particle_name:
                continue

            self.particles.append(particle.Particle())
            self.particles[-1].load_particle(particle_name)
            self.particles[-1].sim_size = self.sim_size
            self.particles[-1].update_measurables(self.dt, self.del_x, self.v_field_total)
        sim_file.close()
        os.chdir("..")






