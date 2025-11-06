from constants import *
import numpy as np
import cmath
#import threading
import yaml
from numba.experimental import jitclass
from numba import types, jit, cuda, njit
import os


class Particle:
    def __init__(self):
        self.prl = 0
        self.pim = 0
        self.E = 0
        self.ke = 0
        self.pe = 0
        self.ptot = 0
        self.sim_size = 0
        self.psi = 0
        self.init_function = None
        self.time_init = False
    
    def initialize(self, init_function):
        self.prl = np.zeros(self.sim_size)
        self.pim = np.zeros(self.sim_size)
        self.init_function = init_function

        for n in range(self.sim_size):
            if init_function.time != None:
                self.time_init = True
                return
            if init_function.real != None:
                self.prl[n] += init_function.real(n)

            if init_function.imag != None:
                self.pim[n] += init_function.imag(n)
            
        ptot = np.sum(self.prl**2 + self.pim**2)
        pnorm = np.sqrt(ptot)
        self.prl = np.divide(self.prl,pnorm)
        self.pim = np.divide(self.pim,pnorm)

    def time_initlize(self, time):

        for n in range(self.sim_size):
            self.prl[n] += self.init_function.time(time)*self.init_function.real(n)
            self.pim[n] += self.init_function.time(time)*self.init_function.imag(n)

    #@cuda.jit('void(float64[:,:], float64[:,:], float64[:], float64[:,:], float64[:], float64[:], float64[:], float64[:,:], float64[:])')
    @jit
    def _fdtd(prl, pim, V, other_particles, ra, rd, sim_size, abc, steps=1):
        # must be static or jit will come and shoot it
        for step in range(steps):
            for n in range(sim_size-1):
                prl[n] += -ra*(pim[n-1] - 2*pim[n] + pim[n+1]) + rd*V[n]*pim[n]
                prl[n] *= abc[n]
                
            for n in range(sim_size-1):
                pim[n] += ra*(prl[n-1] - 2*prl[n] + prl[n+1]) - rd*V[n]*prl[n]
                pim[n] *= abc[n]
        

    def fdtd(self, v_fields, other_particles, ra, rd, abc=None, steps=1):

        if abc is None:
            abc = np.ones(self.sim_size)
            
        Particle._fdtd(self.prl, self.pim, v_fields, other_particles, ra, rd, self.sim_size, abc)
        
        
    def update_measurables(self, dt, dx, V):
        self.psi = self.prl + self.pim*1j
        self.ptot = np.conj(self.psi).dot(self.psi).real
        self.ke = 0        
        for n in range(self.sim_size -1):
            self.ke += (self.psi[n-1] - 2*self.psi[n] + self.psi[n+1])*np.conj(self.psi[n])
        self.ke = (-J2eV*((hbar_J/dx)**2/(2*meff))*self.ke.real)
        self.pe = (np.conj(self.psi)*self.psi).real.dot(V)*J2eV

        self.E = self.pe + self.ke

        return self.psi, self.ptot, self.ke, self.pe, self.E
    
    @jit
    def _run_dft_at_point(prl, pim, point, E, forier, c_time):
        for i in range(len(E)):
            forier[i] += (prl[point] - 1j*pim[point])*cmath.exp(-1j*(2*np.pi*E[i]/h_nobar_eV)*c_time)
        return forier

    def run_dft_at_point(self, E, point, c_time):
        forier = np.zeros(len(E), dtype=complex)
        
        forier = Particle._run_dft_at_point(self.prl, self.pim, point, E, forier, c_time)
        return forier
        
        
    def save_particle(self, filename):
        part_file = open(filename + ".part", "w")

        part_data = {
            "pim" : self.pim,
            "prl" : self.prl
        }
        yaml.dump(part_data, part_file)
        part_file.close()
        
    def load_particle(self, particlename):
        if not os.path.exists(particlename):
            return None
        
        part_file = open(particlename, "r")
        part_data = yaml.load(part_file, Loader=yaml.UnsafeLoader)
        self.pim = part_data["pim"]
        self.prl = part_data["prl"]
        part_file.close()