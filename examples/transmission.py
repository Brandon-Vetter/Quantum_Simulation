import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..") # need to import path above

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
sim_run_time = 10000


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

# setup the input simulation
simin = sim.Simulation(sim_size=sim_size)
simin.init_abc(aabc.Xabc(50))
simin.init_dft(0, 1000, end =.5, loc = dft_point)
simin.init_particle(part.Particle(),init_function.Time_gausian_init(Ein=.27, loc=particle_starting_location))

# run the simulations
print(f"running input simulation (1/2)")
simin.run(steps=sim_run_time)
print("running output simulation (2/2)")
simout.run(steps=sim_run_time)


# graph the simulations
plt.figure()
plt.subplot(2,1,1)
plt.plot(simout.sim_space,simout.v_field_total_eV,'k', label="V_field")
plt.plot(simout.sim_space,simout.particles[0].prl,'b', label="prl")
plt.plot(simout.sim_space,simout.particles[0].pim,'r--', label="pim")
Dft = np.zeros(simout.sim_size)
Dft[quantum.dist_to_step(del_x, dft_point)] = .05
plt.plot(simout.sim_space, Dft, 'k--')
plt.grid()

plt.subplot(2,2,3)
plt.grid()
plt.title("DFT")
plt.plot(simout.dft_E[0], quantum.normalize_dft(simout.dfts[0]), label="Dout")
plt.plot(simin.dft_E[0], quantum.normalize_dft(simin.dfts[0]), label="Din")
plt.xlabel("eV")
plt.legend(bbox_to_anchor=(1.15, .5), loc='upper left')

plt.subplot(2,2,4)
plt.grid()
plt.title("Transmission")
scale = quantum.scale(Vds.drop, simin.dft_E[0])
trans = quantum.transmission(simin.dfts[0], simout.dfts[0])
plt.plot(simin.dft_E[0], trans)
plt.xlabel("eV")
plt.tight_layout()
plt.savefig("transission.png")
plt.show()




