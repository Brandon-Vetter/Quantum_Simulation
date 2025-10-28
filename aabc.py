import numpy as np
from constants import *


class Aabc:
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

    def abc(self, sim_space):
        ret_abc = np.ones(len(sim_space))
        for n in range(int(len(sim_space)/2)):
            if n < self.start_loc:
                ret_abc[n] = self.start_eqt(n)
            if (len(sim_space)-1) - n > len(sim_space) - self.end_loc:
                ret_abc[(len(sim_space)-1) - n] = self.end_eqt(n)
        return ret_abc

        

class Xabc(Aabc):
    """
    abc normally used
    """

    def __init__(self, start, end_offset=0):
        self.start = start
        self.start_eqt = lambda n :1 - .5*((self.start_loc - n)/self.start_loc)**3
        super().__init__(self.start_eqt, start)