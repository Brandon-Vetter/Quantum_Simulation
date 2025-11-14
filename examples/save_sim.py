"""
Demonstration of the simulation saving and loading abilities

time things have not been added yet
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../src") # need to import path above

from constants import *
import simulation as sim
import particle as part
import quantum
import v_field
import init_function
import aabc


# simulation settings
Vgs_value = .5
Vhms_value = .5
Vds_value = 0#.05
dft_point = 340
sim_size = 400
well_size = 100
barrier_size = 6
particle_starting_location = 100
sim_run_time = 500


# setup the output simulation

# setup the V fields
simout = sim.Simulation(sim_size=sim_size)
V = v_field.Well(.5, barrier_size, simout.sim_mid, well_size)
Vgs = v_field.V_pot(simout.sim_mid - well_size/2 - barrier_size, simout.sim_mid + well_size/2 + barrier_size, Vgs_value)
Vhm = v_field.Harmonic(simout.sim_mid - well_size/2, Vhms_value ,end=simout.sim_mid + well_size/2)
Vds = v_field.V_drop(simout.sim_mid - well_size/2 - barrier_size, simout.sim_mid + well_size/2 - barrier_size, -Vds_value)


simout.init_vfield(V)
#simout.init_vfield(Vgs)
simout.init_vfield(Vhm)
#simout.init_vfield(Vds)

# setup the ABC
simout.init_abc(aabc.Xabc(50))

# setup DFT
simout.init_dft(0, 1000, end =.5, loc = dft_point)

# setup particle
simout.init_particle(part.Particle(),init_function.Time_gausian_init(Ein=.27, loc=particle_starting_location))




print("running output simulation")
simout.run(steps=sim_run_time)
simout.save_sim("test_sim")
sim_test = quantum.load_sim("test_sim")

# graph the simulations
plt.figure()
plt.subplot(4,1,1)
plt.plot(sim_test.sim_space,sim_test.v_field_total_eV,'k', label="V_field")

plt.plot(sim_test.sim_space,sim_test.particles[0].prl,'b', label="prl")
plt.plot(sim_test.sim_space,sim_test.particles[0].pim,'r--', label="pim")
Dft = np.zeros(sim_test.sim_size)
Dft[dft_point] = .05
plt.plot(sim_test.sim_space, Dft, 'k--')
plt.grid()
plt.subplot(4,1,2)
plt.plot(simout.sim_space, simout.v_field_total_eV, 'k')
plt.plot(simout.sim_space, simout.particles[0].prl)
plt.plot(simout.sim_space, simout.particles[0].pim)
plt.grid()
plt.subplot(4,1,3)
plt.plot(sim_test.dft_E[0], quantum.normalize_dft(sim_test.dfts[0]))
plt.grid()
plt.ylabel("eV")

plt.subplot(4,1,4)
plt.grid()
plt.plot(simout.dft_E[0], quantum.normalize_dft(simout.dfts[0]))
plt.ylabel("eV")

simout.run(steps=sim_run_time)
sim_test.run(steps=sim_run_time)

plt.figure()
plt.subplot(4,1,1)
plt.plot(sim_test.sim_space,sim_test.v_field_total_eV,'k', label="V_field")

plt.plot(sim_test.sim_space,sim_test.particles[0].prl,'b', label="prl")
plt.plot(sim_test.sim_space,sim_test.particles[0].pim,'r--', label="pim")
Dft = np.zeros(sim_test.sim_size)
Dft[dft_point] = .05
plt.plot(sim_test.sim_space, Dft, 'k--')
plt.grid()
plt.subplot(4,1,2)
plt.plot(simout.sim_space, simout.v_field_total_eV, 'k')
plt.plot(simout.sim_space, simout.particles[0].prl)
plt.plot(simout.sim_space, simout.particles[0].pim)
plt.grid()
plt.subplot(4,1,3)
plt.plot(sim_test.dft_E[0], quantum.normalize_dft(sim_test.dfts[0]))
plt.grid()
plt.ylabel("eV")

plt.subplot(4,1,4)
plt.grid()
plt.plot(simout.dft_E[0], quantum.normalize_dft(simout.dfts[0]))
plt.ylabel("eV")
