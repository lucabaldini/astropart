# Copyright (C) 2023, Luca Baldini (luca.baldini@pi.infn.it)
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

"""Basic atmospheric characteristics.
"""

import numpy as np


R = 8.31446261815324 # J s2 / (K m mol)
g = 9.80665 # m/s
c = 299792458. # m / s
m_pi = 139.57 # MeV
tau_pi = 2.6e-8 # s
m_pi0 = 134.98 # MeV
m_mu = 104.66 # MeV
tau_mu = 2.197e-6 # s


class Atmosphere:

    MOLAR_MASS = 28.95e-3 # kg / mol
    DENSITY_ASL = 1.293e-3 # g / cm^3
    PRESSURE_ASL = 1013.25 # mbar
    REFRACTION_INDEX_ASL = 1.000293
    ALPHA_GLADSTONE_DALE = 0.236
    AVERAGE_STRATOSPHERE_TEMPERATURE = 240. # deg K
    SCALE_HEIGHT = 1.e-3 * R * AVERAGE_STRATOSPHERE_TEMPERATURE / g / MOLAR_MASS # km
    TOTAL_GRAMMAGE = 1.e5 * DENSITY_ASL * SCALE_HEIGHT # g / cm^2
    RADIATION_LENGTH = 37. # g / cm^2
    PROTON_INTERACTION_LENGTH = 90. # g / cm^2
    PION_INTERACTION_LENGTH = 1.5 * PROTON_INTERACTION_LENGTH

    @staticmethod
    def _exp_scaling(altitude: float) -> float:
        """Return the exponential scaling that applies, e.g., to the pressure and
        density.
        """
        return np.exp(-altitude / Atmosphere.SCALE_HEIGHT)

    @staticmethod
    def pressure(altitude: float) -> float:
        """Return the pressure (in mbar) at a given altitude.
        """
        return Atmosphere.PRESSURE_ASL * Atmosphere._exp_scaling(altitude)

    @staticmethod
    def density(altitude: float) -> float:
        """Return the density (in g / cm^3) at a given altitude.
        """
        return Atmosphere.DENSITY_ASL * Atmosphere._exp_scaling(altitude)

    @staticmethod
    def grammage(altitude: float) -> float:
        """Return the residual grammage (in g / cm^2) above a given altitude.
        """
        return Atmosphere.TOTAL_GRAMMAGE * Atmosphere._exp_scaling(altitude)

    @staticmethod
    def radiation_length_to_altitude(depth: float) -> float:
        """Calculate the altitude at which the corresponding grammage is equivalent
        to a given depth in terms of radiation lengths.
        """
        C = Atmosphere.TOTAL_GRAMMAGE / depth / Atmosphere.RADIATION_LENGTH
        return Atmosphere.SCALE_HEIGHT * np.log(C)

    @staticmethod
    def proton_interaction_length_to_altitude(depth: float) -> float:
        """Calculate the altitude at which the corresponding grammage is equivalent
        to a given depth in terms of proton interaction lengths.
        """
        C = Atmosphere.TOTAL_GRAMMAGE / depth / Atmosphere.PROTON_INTERACTION_LENGTH
        return Atmosphere.SCALE_HEIGHT * np.log(C)

    @staticmethod
    def refraction_index(altitude: float) -> float:
        """Return the refraction index at a given altitude.
        """
        density = Atmosphere.density(altitude)
        return 1. + Atmosphere.ALPHA_GLADSTONE_DALE * density

    def cherenkov_energy_threshold(altitude: float, mass: float) -> float:
        """Return the Cherenkov energy threshold (in MeV) at a given altitude
        """
        density = Atmosphere.density(altitude)
        return mass / np.sqrt(2. * Atmosphere.ALPHA_GLADSTONE_DALE * density)

    def cherenkov_angle(altitude: float) -> float:
        """Return the Cherenkov angle (in decimal degrees) at a given altitude.
        """
        density = Atmosphere.density(altitude)
        return np.degrees(np.sqrt(2. * Atmosphere.ALPHA_GLADSTONE_DALE * density))

    def pion_decay_energy_threshold(altitude: float) -> float:
        """
        """
        density = Atmosphere.density(altitude)
        return 1.e-2 * m_pi * Atmosphere.PION_INTERACTION_LENGTH / c / tau_pi / density

    def muon_decay_energy_threshold(altitude: float) -> float:
        """
        """
        return 1.e3 * m_mu * altitude / c / tau_mu


print(Atmosphere.pion_decay_energy_threshold(8.))
print(Atmosphere.muon_decay_energy_threshold(20.))
