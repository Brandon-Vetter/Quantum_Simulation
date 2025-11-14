import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append("../src") # need to import path above
import quantum
import simulation
import particle
import v_field
import init_function

sim = simulation.Simulation(dt=8E-17, del_x=.2E-9, sim_size=200)
sim.init_dft(0, 1000, end=.5)
time_ip = init_function.Time_gausian_init(Ein=.27, loc=100)

part = particle.Particle()
sim.init_particle(part, time_ip)

while True:
    try:
        steps = int(input("time steps: "))
    except ValueError:
        continue
    
    sim.run(steps=steps)
    
    plt.plot(sim.sim_space, sim.particles[0].prl)
    plt.plot(sim.sim_space, sim.particles[0].pim)
    plt.grid()
    plt.show()

