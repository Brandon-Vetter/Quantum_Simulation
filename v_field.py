import numpy as np
from constants import *



class Vfield:
    """
    Base class for potential fields in the simulation
    """
    sim_size = 0
    del_x = 0
    def __init__(self, eqt = None, *args):
        self.eqt = eqt
        self.v_field = None
        self.args = args

    def init_eqt(self):
        self.v_field = self.eqt(self.sim_size, *self.args)

    def step(self, time):
        pass


class Barrier(Vfield):
    def __init__(self, height, length, loc, spatial=False):
        eqt = lambda n, l, h, p : np.array([h*eV2J if (i >= p-int(l/2) and i <= p+int(l/2)) 
                                            else 0.0 for i in range(n)])
        self.spatial = spatial
        self.loc = loc
        self.height = height
        self.length = length
        super().__init__(eqt)

    def init_eqt(self):
        if self.spatial:
            self.loc = int(self.loc/self.del_x)
            self.length = int(self.length/self.del_x)
        self.v_field = self.eqt(self.sim_size, self.length, self.height, self.loc)


class Well(Barrier):
    def __init__(self, height, length, loc, well_size, spatial=False):
        super().__init__(height, length, loc, spatial)
        self.well_size = well_size

    def init_eqt(self):
        if self.spatial:
            self.well_size = self.del_x*self.well_size
            self.loc = int(self.loc/self.del_x)
            self.length = int(self.length/self.del_x)

        hwell = int(self.well_size/2)
        self.v_field = self.eqt(self.sim_size, self.length, self.height, self.loc - hwell - int(self.length/2))
        self.v_field += self.eqt(self.sim_size, self.length, self.height, self.loc + hwell + int(self.length/2))


class V_pot(Vfield):
    def __init__(self, start, end, value, spatial=False):
        eqt = lambda n, s, e, v: [v if i >= s and i <= e else 0 for i in range(n)]
        self.start = start
        self.end = end
        self.value = value*eV2J
        self.spatial = spatial
        super().__init__(eqt)
    
    def init_eqt(self):
        if self.spatial:
            self.start = int(self.start/self.del_x)
            self.end = int(self.end/self.del_x)
        
        self.v_field = self.eqt(self.sim_size, self.start, self.end, self.value)
    

class V_drop(Vfield):
    def __init__(self, start, end, drop, spatial=False):
        super().__init__(self.eqt)
        self.start=start
        self.end=end
        self.drop = drop*eV2J
        self.spatial=spatial

    def init_eqt(self):
        if self.spatial:
            self.start = int(self.start/self.del_x)
            self.end = int(self.end/self.del_x)
        
        slope = self.drop/(self.end - self.start)
        self.v_field = self.eqt(self.sim_size, self.start, self.end, slope, self.drop)

    def eqt(self, n, start, end, slope, drop):
        ind = 0
        ret_arr = np.zeros(n)
        for i in range(n):
            if i >= start and i <= end:
                ret_arr[i] = slope*ind
                ind+=1
            elif i > end:
                ret_arr[i] = drop
        
        return ret_arr
    

class Harmonic(Vfield):
    def __init__(self, start, drop, size=0, end=0, spatial=False):

        self.start = start
        self.end = end
        self.drop = drop*eV2J
        self.size = size
        self.mid = self.start + (self.end - self.start)/2
        self.spatial = spatial
        eqt = lambda n, s, m, e, d, sl: [sl*(i-m)**2 - d if i >= s and i <= e else 0 for i in range(n)]
        super().__init__(eqt)

    def init_eqt(self):
        if self.spatial:
            self.end = int(self.end/self.del_x)
            self.start = int(self.start/self.del_x)
            self.size = int(self.size/self.del_x)
            self.mid = int(self.mid/self.del_x)
        
        self.mid = int(self.mid)
        if self.end == 0:
            self.mid = self.start
            self.start = int(self.mid - self.size/2)
            self.end = self.start + self.size
        else:
            self.size = self.end - self.start
        slope = self.drop/(self.end - self.mid)**2

        self.v_field = self.eqt(self.sim_size, self.start, self.mid, self.end, self.drop, slope)


class Vwell(V_drop):
    def __init__(self, start, drop, size = 0, end=0, spatial=False):
        self.size = size
        self.mid = start + (end - start)/2
        super().__init__(start, end, drop, spatial)

    def init_eqt(self):
        if self.spatial:
            self.end = int(self.end/self.del_x)
            self.start = int(self.start/self.del_x)
            self.size = int(self.size/self.del_x)
            self.mid = int(self.mid/self.del_x)
        
        self.mid = int(self.mid)
        if self.end == 0:
            self.mid = self.start
            self.start = int(self.mid - self.size/2)
            self.end = self.start + self.size
        else:
            self.size = self.end - self.start
        slope = self.drop/(self.end - self.mid)
        self.v_field = self.eqt(self.sim_size, self.start, self.mid, -self.slope)
        self.v_field += self.eqt(self.sim_size, self.mid, self.end, self.slope)