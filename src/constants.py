"""quantum.constants

Physical and simulation constants used throughout the project.

:filename: constants.py
:author: Brandon Vetter <brandon.w.vetter@gmail.com>
:license: Apache License 2.0 <https://www.apache.org/licenses/LICENSE-2.0>
:summary: Centralized physical constants and semiconductor effective masses
    used by the simulations.

This module centralizes commonly used physical constants (with units)
and a few semiconductor effective-mass values used by the simulations.

Constants
---------
- ``hbar_J``: reduced Planck constant in J*s
- ``hbar_eV``: reduced Planck constant in eV*s
- ``h_nobar_eV``: Planck constant (not reduced) in eV*s
- ``h_nobar_J``: Planck constant (not reduced) in J*s
- ``m0``: electron rest mass in kg
- ``meff``: effective mass used in simulations (defaults to ``m0``)
- ``ecoul``: elementary charge in Coulombs
- ``epsz``: vacuum permittivity (in SI units)
- ``eV2J``: factor to convert eV -> Joules
- ``J2eV``: factor to convert Joules -> eV
- ``G0``: quantum conductance in microSiemens

Semiconductor effective masses (relative to ``m0``)
 - ``Si``: silicon effective mass (m*/m0)
 - ``GaAs``: gallium-arsenide effective mass (m*/m0)
 - ``Ge``: germanium effective mass (m*/m0)

Note: numeric values are approximate and chosen for quick simulations and
examples; use higher-precision or library-provided constants for production
calculations when needed.
"""

from enum import Enum

######################
# Global Varables
#####################

# Quantum Varables

hbar_J = 1.054e-34 # J*s
hbar_eV = 6.58e-16 # eV
h_nobar_eV = 4.135e-15  # in eV
h_nobar_J = 6.625e-34 # J*s
m0 = 9.1e-31 # kg
meff = m0 
ecoul = 1.6e-19 # C
epsz = 8.85e-9 # H/m permebilit of free space
eV2J = 1.6e-19 # Convert to Joules
J2eV = 1/eV2J # Convert to eV
G0 = 38.7 # micro S Quantum Conductance


# Semiconductor Values
    # all in m/me
Si = 1.08
GaAs = 0.067
Ge = 0.55


