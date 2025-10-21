from constants import *
import numpy as np
from numba.experimental import jitclass
from numba import types, jit


class particle:
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
        self.forour = False
    
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

    @jit
    def _fdtd(prl, pim, V, other_particles, ra, rd, sim_size, abs):
        # must be static or jit will come and shoot it
        for n in range(sim_size-1):
            prl[n] += -ra*(pim[n-1] - 2*pim[n] + pim[n+1]) + rd*V[n]*pim[n]
            prl[n] *= abs[n]
            
        for n in range(sim_size-1):
            pim[n] += ra*(prl[n-1] - 2*prl[n] + prl[n+1]) - rd*V[n]*prl[n]
            pim[n] *= abs[n]

    def fdtd(self, v_fields, other_particles, ra, rd, abs=None):

        if abs == None:
            abs = np.ones(self.sim_size)
            
        particle._fdtd(self.prl, self.pim, v_fields, other_particles, ra, rd, self.sim_size, abs)
        
        
    def update_measurables(self, dt, dx, V):
        self.psi = self.prl + self.pim*1j
        self.ptot = np.conj(self.psi).dot(self.psi).real
        self.ke = 0        
        for n in range(self.sim_size -1):
            self.ke += (self.psi[n-1] - 2*self.psi[n] + self.psi[n+1])*np.conj(self.psi[n])
        self.ke = (-J2eV*((hbar_J/dx)**2/(2*meff))*self.ke.real)
        self.pe = (np.conj(self.psi)*self.psi).real.dot(V)*J2eV

        self.E = self.pe + self.ke



class init_function:
    location = None
    del_x = 0
    dt = 0
    real = None
    imag = None
    t = None
    time = None
    def __init__(self):
        pass
    
    def real(self, n):
        return 0
    
    def imag(self, n):
        return 0


class gausian_init(init_function):
    def __init__(self, lambd, sigma, loc):
        super().__init__()
        self.location = loc
        self.real = lambda n : np.exp(-1.*((n-self.location)/sigma)**2)*np.cos(2*np.pi*(n-self.location)/lambd)
        self.imag = lambda n : np.exp(-1.*(((n-self.location)/sigma)**2))*np.sin(2*np.pi*(n-self.location)/lambd)


class time_gausian_init(init_function):
    def __init__(self, lambd, sigma, loc, Ein, time=0, steps=0):
        super().__init__()
        self.location = loc
        self.lambd = lambd
        self.sigma = sigma
        self.E = Ein
        self.real = self.real


    def real(self, n):
        if n == self.location:
            
            return .05
        else:
            return 0
    
    def time(self, c_time):
        T_per = (h_nobar_J/(self.E*self.dt))
        sig = .67*T_per
        T = 2*sig
        return np.exp(-1.*((c_time - T)/sig)**2)*np.cos(2*np.pi*(c_time - T)/T_per)