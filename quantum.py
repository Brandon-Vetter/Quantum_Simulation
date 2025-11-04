###############################################################################
#
# quantum.py
# 
# @Function: contains all the functions used for quantum mechanics
#
###############################################################################

from numba import jit
import numpy as np
import simulation as sim

def pml(NN, cells = 50):
    nabc = cells
    xabc = np.ones(NN)
    show_pml = np.zeros(NN)
    for n in range(nabc,NN-nabc):
        show_pml[n] = 1

    for n in range(nabc + 1):
        xxn = (nabc - n) / nabc
    #    xpml[n] = 1 - .25 * xxn ** 3
        xabc[n] = 1 - .5 * xxn ** 3
        xabc[-n] = xabc[n]    
    return xabc

# fdtd simulations
@jit(parallel=True)
def fdtd(prl, pim, n_step, ra, rd, V=None, pml=None, pml_args = []):
    NN = len(prl)
    if V == None:
        V = np.zeros(NN)
    if pml == None:
        abc = np.ones(NN)
    else:
        abc = pml(*pml_args)
    print("FDTD: nstep = ",n_step)
    for _ in range(n_step):
        prl[1] = abc[1]*prl[1] - ra*(pim[0] - 2*pim[1] + pim[2]) + rd*V[1]*pim[1]
        for n in range(2, NN-2):
            prl[n] = abc[n]*prl[n] - ra*(pim[n-1] - 2*pim[n] + pim[n+1]) + rd*V[n]*pim[n]
            pim[n-1] = abc[n-1]*pim[n-1] + ra*(prl[n-2] - 2*prl[n-1] + prl[n]) - rd*V[n-1]*prl[n-1]
            
        pim[NN-2] = abc[NN-2]*pim[NN-2] + ra*(prl[NN-3] - 2*prl[NN-2] + prl[NN-1]) - rd*V[NN-2]*prl[NN-2]

    return prl, pim

def normalize_dft(dft):
    return np.abs(np.conj(dft)*dft)

def scale(V_drop, E_dft):
    return np.sqrt((E_dft + V_drop)/E_dft)

def transmission(input, output, scale=None):
    if scale == None:
        scale = np.ones(len(input))
    
    return np.abs(output/input)*scale

def load_sim(simname):
    sim_ret = sim.Simulation()
    sim_ret.load_sim(simname)
    return sim_ret

def time_to_step(dt, time):
    return int(time/dt)

def step_to_time(dt, step):
    return dt*step

def dist_to_step(del_x, dist):
    return int(dist/del_x)

def step_to_dist(del_x, step):
    return del_x*step

def delta(x, loc, height):
    y = np.zeros(len(x))
    y[loc] = height

def calulate_mod(start, end, particle):
    psi = particle.prl + 1j*particle.pim
    ind = start
    mod = 0
    while ind < end:
        mod += psi[ind].conj*psi[ind]
        ind += 1

    return mod
