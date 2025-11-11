
"""quantum.particle

Classes and utilities for particle representations used in simulations.

:filename: particle.py
:author: Brandon Vetter <brandon.w.vetter@gmail.com>
:license: Apache License 2.0 <https://www.apache.org/licenses/LICENSE-2.0>
:summary: Code to manage the particles in the simulation


This module defines the :class:`Particle` class.  The 'Particle' class is 
what manages where the paricle is in the simulation, and how it responds
to the eletric potentials (V-fields).  This is where the FDTD for the
shodinger equation is ran for the simulation.
"""

from constants import *
import numpy as np
import cmath
#import threading
import yaml
from numba.experimental import jitclass
from numba import types, jit, cuda, njit
import os


class Particle:
    """
    Particle representing a single quantum wavefunction on a 1D grid.

    This class stores the real and imaginary parts of a wavefunction and
    provides routines to initialize it, evolve is using FDTD,
    compute measurable quantities (kinetic/potential energy,
    total probability), and save/load the particle state.

    :ivar prl: real part of the wavefunction (numpy array or scalar)
    :ivar pim: imaginary part of the wavefunction (numpy array or scalar)
    :ivar E: total energy (scalar)
    :ivar ke: kinetic energy (scalar)
    :ivar pe: potential energy (scalar)
    :ivar ptot: total probability / norm (scalar)
    :ivar sim_size: spatial grid size (int)
    :ivar psi: complex-valued wavefunction array
    :ivar init_function: initialization object with `real`, `imag`, and optionally `time` callables
    :ivar time_init: True if initialization uses a time-dependent factor
    """
    def __init__(self):
        """
        Create a new Particle and initialize numeric fields.

        :returns: None
        """
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
        """
        Initialize the particle's real and imaginary arrays from an
        initialization function and normalize the resulting wavefunction.

        :param init_function: object providing `real(n)`, `imag(n)`, and
            optionally `time(t)` to build the initial state
        :returns: None
        """
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
        """
        Apply a time-dependent factor from the stored init_function to the
        real and imaginary parts of the particle.

        :param time: scalar time value used by the init_function.time method
        :returns: None
        """

        for n in range(self.sim_size):
            self.prl[n] += self.init_function.time(time)*self.init_function.real(n)
            self.pim[n] += self.init_function.time(time)*self.init_function.imag(n)

    #@cuda.jit('void(float64[:,:], float64[:,:], float64[:], float64[:,:], float64[:], float64[:], float64[:], float64[:,:], float64[:])')
    @jit
    def _fdtd(prl, pim, V, other_particles, ra, rd, sim_size, abc, steps=1):
        # must be static or jit will come and shoot it
        """
        Low-level finite-difference time-domain update for the wavefunction.

        Note: this is a static/jit-compatible function that operates in-place on
        the provided arrays.

        The Schrödinger equation is broken into its real and imaginary parts.  It is
        then calulated using the FDTD equation:

        .. math::
            \\psi_{real}^{t} = \\psi_{real}^{t-1} - ra(\\psi_{imag}(t, n-1) - 2 \\cdot \\psi_{imag}(t, n) + \\psi_{imag}(t, n+1)) + rd(V(t, n) \\cdot \\psi_{imag})
        Real part of the shrodinger equation

        .. math::
            \\psi_{imag}^{t} = \\psi_{imag}^{n, t-1} + ra(\\psi_{real}(t, n-1) - 2 \\cdot \\psi_{real}(t, n) - \\psi_{real}(t, n+1)) + rd(V(t, n) \\cdot \\psi_{real})
        Imaginary part of the shrodinger equation

        Where ra and rd are constants defined when :math:`\Delta x` and :math:`\\Delta t` are chosen.  See simulation init for how ra and rd are setup.

        :param prl: numpy array of real parts of the shrodinger equation
        :param pim: numpy array of imaginary parts of the shrodinger equation
        :param V: potential array
        :param other_particles: unused/placeholder for interactions with other particles
        :param ra: coefficient for kinetic term update
        :param rd: coefficient for potential term update
        :param sim_size: integer simulation size / array length
        :param abc: absorbing boundary coefficient array (same length as arrays)
        :param steps: number of time steps to perform (default 1)
        :returns: None (arrays are updated in-place)
        """

        for step in range(steps):
            for n in range(sim_size-1):
                prl[n] = abc[n]*prl[n] - ra*(pim[n-1] - 2*pim[n] + pim[n+1]) + rd*V[n]*pim[n]
                
            for n in range(sim_size-1):
                pim[n] = abc[n]*pim[n] + ra*(prl[n-1] - 2*prl[n] + prl[n+1]) - rd*V[n]*prl[n]
        

    def fdtd(self, v_fields, other_particles, ra, rd, abc=None, steps=1):
        """
        Public wrapper for the finite-difference time-domain update.

        :param v_fields: array-like potential values
        :param other_particles: placeholder or list of other particles
        :param ra: coefficient used for kinetic updates
        :param rd: coefficient used for potential updates
        :param abc: absorbing boundary coefficient array; if None, uses ones
        :param steps: number of time steps to run (currently forwarded to inner function)
        :returns: None
        """

        if abc is None:
            abc = np.ones(self.sim_size)
            
        Particle._fdtd(self.prl, self.pim, v_fields, other_particles, ra, rd, self.sim_size, abc)
        
        
    def update_measurables(self, dt, dx, V):
        """
        Calculate and update measurable physical quantities for the particle.

        :param dt: time step (not used directly here but kept for API compatibility)
        :param dx: spatial discretization step used for kinetic energy calculation
        :param V: potential array used to compute potential energy
        :returns: tuple (psi, ptot, ke, pe, E)
            - psi: complex wavefunction array
            - ptot: total probability (norm)
            - ke: kinetic energy (scalar)
            - pe: potential energy (scalar)
            - E: total energy (scalar)
        """

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
        """
        Increment a provided Fourier array with the contribution from a single
        spatial point at a set time (DFT slice).  Is static to be jit compatable.

        Runs the following equation on the particle at the specified point for every time step:

        .. math::
            F(E,t) = F(E,t-1) + \psi(point, t) \cdot e^{-i \frac{2 \pi E}{\hbar} t}

        where :math:`\psi = \psi_{real} - j\psi_{img}`
        and t would just be the current simulation step and point is the location of
        where the DFT is taken.
        
        :param prl: real-part of the shrodinger equation
        :param pim: imag-part imaginary part of the shrodinger equation
        :param point: integer index of the spatial point to sample
        :param E: array of energies/frequencies to evaluate
        :param forier: complex array to accumulate DFT results (in-out)
        :param c_time: scalar time at which to sample
        :returns: forier (the updated complex accumulation array)
        """
        for i in range(len(E)):
            forier[i] += (prl[point] - 1j*pim[point])*cmath.exp(-1j*(2*np.pi*E[i]/h_nobar_eV)*c_time)
        return forier

    def run_dft_at_point(self, E, point, c_time):
        """
        Wrapper for the static _run_dft_at_point function.

        :param E: array of energies/frequencies to evaluate
        :param point: integer index of the spatial point to sample
        :param c_time: scalar time at which to sample
        :returns: forier, complex numpy array of DFT results (length=len(E))
        """
        forier = np.zeros(len(E), dtype=complex)
        
        forier = Particle._run_dft_at_point(self.prl, self.pim, point, E, forier, c_time)
        return forier
        
        
    def save_particle(self, filename):
        """
        Save the particle's real and imaginary arrays to a YAML-based .part file.

        :param filename: base filename (without .part extension) to write
        :returns: None
        """
        part_file = open(filename + ".part", "w")

        part_data = {
            "pim" : self.pim,
            "prl" : self.prl
        }
        yaml.dump(part_data, part_file)
        part_file.close()
        
    def load_particle(self, particlename):
        """
        Load particle data from a .part YAML file into this Particle instance.

        :param particlename: path to the .part file to load
        :returns: None if the file does not exist; otherwise None but populates
                  self.prl and self.pim with loaded arrays
        """
        if not os.path.exists(particlename):
            return None
        
        part_file = open(particlename, "r")
        part_data = yaml.load(part_file, Loader=yaml.UnsafeLoader)
        self.pim = part_data["pim"]
        self.prl = part_data["prl"]
        part_file.close()