# Copyright (C) 2024, Luca Baldini (luca.baldini@pi.infn.it)
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""Neutrino mixing.
"""

from enum import IntEnum

import numpy as np
import astropy.constants as const
import astropy.units as u

# Current best estimates based on nufit v5.2, http://www.nu-fit.org/?q=node/256
# assuming normal ordering and including the SK atmospheric data.
# Note that, for the sake of simplicity, we assume a trivial CP-violating phase,
# as the latter is not really constrained by current experiments.
THETA_12_CBE = 33.41
THETA_23_CBE = 42.2
THETA_13_CBE = 8.58
DELTA_CP_CBE = 232.

DELTA_M21_CBE = 7.41e-5
DELTA_M32_CBE = 2.507e-3

# This is the famous 1.27 that is found in the neutrino oscillation formula.
_PHASE_PREFACTOR = (1. / 4. / const.hbar / const.c).to_value(u.MeV / u.m / u.eV**2.)

# Fermi constant from https://physics.nist.gov/cgi-bin/cuu/Value?gf
# Note we are caching the value in eV cm^3.
G_F = (1.6663787e-5 * u.GeV**-2. * (const.hbar * const.c)**3.).to_value(u.eV * u.cm**3.)




class MassEigenstate(IntEnum):

    """Enum class for the neutrino mass eigestates.
    """
    ONE = 1
    TWO = 2
    THREE = 3



class FlavorEigenstate(IntEnum):

    """Enum class for the neutrino flavor eigenstates.
    """
    ELECTRON = 1
    MUON = 2
    TAU = 3



class MixingMatrix(np.ndarray):

    """Class describing a 3 x 3 mixing matrix.
    """

    def __new__(cls, theta12: float = THETA_12_CBE, theta23: float = THETA_23_CBE,
        theta13: float = THETA_13_CBE, delta_cp: float = DELTA_CP_CBE) -> None:
        """Constructor.
        """
        # Calculate the trigonometric functions.
        s12 = np.sin(np.radians(theta12))
        c12 = np.cos(np.radians(theta12))
        s23 = np.sin(np.radians(theta23))
        c23 = np.cos(np.radians(theta23))
        s13 = np.sin(np.radians(theta13))
        c13 = np.cos(np.radians(theta13))
        ph = np.exp(1j * np.radians(delta_cp))
        # And here is the glorious PMNS matrix.
        data = (
            (c12 * c13, s12 * c13, s13 / ph),
            (-s12 * c23 - c12 * s13 * s23 * ph, c12 * c23 - s12 * s13 * s23 * ph, c13 * s23),
            (s12 * s23 - c12 * s13 * c23 * ph, -c12 * s23 - s12 * s13 * c23 * ph, c13 * c23)
        )
        return super().__new__(cls, (3, 3), complex, np.array(data))

    def flavor_content(self, mass_eigenstate: MassEigenstate) -> np.ndarray:
        """Return the flavor content in a given mass eigenstate.
        """
        vec = self.T[mass_eigenstate - 1]
        return np.real(vec * vec.conjugate())





def oscillation_length(energy: float, delta_m: float) -> float:
    """Return the oscillation length (in m) for a given energy (in MeV) and
    difference of mass squared (in eV^2).
    """
    return energy / delta_m / np.pi / _PHASE_PREFACTOR


def oscillation_length_21(energy: float) -> float:
    """Return the oscillation length (in m) for a given energy (in MeV) assuming
    the current best estimate for delta m^2_21.
    """
    return energy / DELTA_M21_CBE / np.pi / _PHASE_PREFACTOR


def oscillation_length_32(energy: float) -> float:
    """Return the oscillation length (in m) for a given energy (in MeV) assuming
    the current best estimate for delta m^2_32.
    """
    return energy / DELTA_M32_CBE / np.pi / _PHASE_PREFACTOR


def msw_resonance_energy(electron_density: float, delta_m: float = DELTA_M21_CBE,
    theta_mixing: float = THETA_12_CBE) -> float:
    """Return the MSW resonance energy for a given set of parameters.
    """
    return delta_m * np.cos(2. * np.radians(theta_mixing)) / 2. / np.sqrt(2.) / G_F / electron_density




if __name__ == '__main__':
    m = MixingMatrix()
    print(abs(m))
    print()
    print(abs(m)**2.)
    print()
    for eigenstate in MassEigenstate:
        print(f'{eigenstate} -> {m.flavor_content(eigenstate)}')
