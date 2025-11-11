import numpy as np
from constants import *


class Init_function:
    location = None
    del_x = 0
    dt = 0
    real = None
    imag = None
    t = None
    time = None
    def __init__(self, loc=0, pos=0):
        self.loc = loc
        self.pos = pos
    
    def initialize(self):
        if self.loc == 0:
            self.location = round(self.pos/self.del_x)
        else:
            self.location = self.loc

    def real(self, n):
        return 0
    
    def imag(self, n):
        return 0


class Gausian_init(Init_function):
    def __init__(self, lambd, sigma, loc=0, pos=0):
        super().__init__(loc, pos)
        self.real = lambda n : np.exp(-1.*((n-self.location)/sigma)**2)*np.cos(2*np.pi*(n-self.location)/lambd)
        self.imag = lambda n : np.exp(-1.*(((n-self.location)/sigma)**2))*np.sin(2*np.pi*(n-self.location)/lambd)


class Time_gausian_init(Init_function):
    def __init__(self, loc=0, pos=0, lambd=0, sigma=0, Ein=0, pulse_length=.65, time=0, steps=0):
        super().__init__(loc, pos)
        
        self.lambd = lambd
        self.sigma = sigma
        self.E = Ein
        self.start_time = time
        self.pulse_length = pulse_length
        self.real = self.real

    def real(self, n):
        if n == self.location:
            return .05
        else:
            return 0
    
    def time(self, c_time):
        if self.lambd == 0:
            self.lambd = (h_nobar_eV/(self.E))
        
        if self.sigma == 0:
            self.sigma = self.pulse_length*self.lambd

        if self.start_time == 0:
            self.start_time = 2.*self.sigma
        
        
        return np.exp(-1.*((c_time - self.start_time)/self.sigma)**2)*np.cos((2*np.pi*(c_time - self.start_time))/self.lambd)