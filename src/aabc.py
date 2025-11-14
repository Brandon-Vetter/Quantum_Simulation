import numpy as np
from constants import *


class Aabc:
    """
    adds the barrier at the start and end of the sim to remove reflections
    """
    del_x = 0
    sim_space = None
    def __init__(self, eqt_s, start, end_offset=0, eqt_e=None, spatial=False):
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

        self.spatial = spatial

    def abc(self, sim_space):
        if self.spatial:
            self.start_loc = round(self.start_loc/self.del_x)
            self.end_loc = round(self.end_loc/self.del_x)


        ret_abc = np.ones(len(sim_space))
        for n in range(round(len(sim_space)/2)):
            if n < self.start_loc:
                ret_abc[n] = self.start_eqt(n, self.start_loc)
            if (len(sim_space)-1) - n > len(sim_space) - self.end_loc:
                ret_abc[(len(sim_space)-1) - n] = self.end_eqt(n, self.end_loc)
        return ret_abc

        

class Xabc(Aabc):
    """
    abc normally used
    """

    def __init__(self, start, end_offset=0, dec=.5, spatial=False):
        self.start = start
        self.start_eqt = lambda n, s:1 - dec*((s - n)/s)**3
        super().__init__(self.start_eqt, start, spatial=spatial)