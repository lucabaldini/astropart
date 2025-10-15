# Copyright (C) 2025, Luca Baldini (luca.baldini@pi.infn.it)
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

"""White dwarfs.
"""

from astropy import constants as const
from astropy import units as u
from astropy.units.quantity import Quantity
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from astropart.qstat import DegenerateFermiGas


class WhiteDwarfBase:

    """Base class for white dwarfs.
    """

    REFERENCE_MU = 2.


    @staticmethod
    def interpolate_radius(mass_grid: Quantity, radius_grid: Quantity, mass: Quantity) -> Quantity:
        """Interpolate the radius of a white dwarf for a given mass.

        Arguments
        ---------
        mass_grid : Quantity (mass)
            The grid of masses.

        radius_grid : Quantity (length)
            The grid of radii.

        mass : Quantity (mass)
            The mass of the star.
        """
        x = mass_grid / const.M_sun
        y = radius_grid.to(u.km).value
        spline = InterpolatedUnivariateSpline(x, y, k=3, ext='zeros')
        return spline(mass / const.M_sun) * u.km


class UniformWhiteDwarf:

    """Small class describing a (fictional) white dwarf, modeled as a uniform sphere.

    This is largely based on E. C. Stoner, "The Equilibrium of Dense Stars",
    Phil. Mag. 9, 944 (1930), with most of the formulae in the paper re-implemented
    verbatim in this class.
    """

    REFERENCE_MU = WhiteDwarfBase.REFERENCE_MU

    @staticmethod
    def gravitational_energy_density(M: Quantity, n: Quantity, mu: float = REFERENCE_MU) -> Quantity:
        """Return the gravitational energy density for a uniform sphere of given
        characteristics.

        Arguments
        ---------
        M : Quantity (mass)
            The total mass of the star.

        n : Quantity (number density)
            The electron number density.

        mu : float
            The average number of nucleons per electron.
        """
        value = -3. / 5. * (4. / 3.)**(1. / 3.) * const.G * (np.pi * mu * const.m_p)**(4. / 3.) * \
                M**(2. / 3.) * n**(4. / 3.)
        return value.to(u.eV / u.cm**3)

    @staticmethod
    def non_relativistic_fermi_energy_density(n: Quantity) -> Quantity:
        """Return the Fermi degeneracy energy density in the non-relativistic limit.

        Arguments
        ---------
        n : Quantity (number density)
            The electron number density.
        """
        value = (const.h**3 / 8. / np.pi)**(2. / 3.) * 3**(5. / 3.) / 10. / \
                const.m_e * n**(5. / 3.)
        return value.to(u.eV / u.cm**3)

    @staticmethod
    def non_relativistic_stoner_number_density(M: Quantity, mu: float = REFERENCE_MU) -> Quantity:
        """Return the equilibrium number density for a uniform sphere derived in
        Stoner (1930).

        Arguments
        ---------
        M : Quantity (mass)
            The total mass of the star.

        mu : float
            The average number of nucleons per electron.
        """
        value = 256 * (np.pi / 3.)**3. * const.G**3. * mu**4. * const.m_p**4. * \
                const.m_e**3. / const.h**6. * M**2.
        return value.to(u.cm**-3)

    @staticmethod
    def non_relativistic_stoner_radius(M: Quantity, mu: float = REFERENCE_MU) -> Quantity:
        """Return the equilibrium radius for a uniform sphere derived in Stoner (1930)
        in the non relativistic limit. Note this is only approxmiately valid for
        small WD masses (say 0.1 solar masses or less).

        Arguments
        ---------
        M : Quantity (mass)
            The total mass of the star.

        mu : float
            The average number of nucleons per electron.
        """
        value = 2**(-10. / 3.) * (3. / np.pi)**(4. / 3.) * const.h**2. / const.G /\
                mu**(5. / 3.) / const.m_p**(5. / 3.) / const.m_e * M**(-1. / 3)
        return value.to(u.km)

    @staticmethod
    def _F(x: float) -> float:
        """Implementation of equation (19) in Stoner (1930).

        This is used to get the inverse formula for number density at the equilibrium.

        Arguments
        ---------
        x : float
            The (adimensional) normalized Fermi momentum p_F / (m_e c).
        """
        return 1. / (8. * x**3.) * (3. / x * np.log(x + np.sqrt(1. + x**2.)) + \
            np.sqrt(1. + x**2) * (2. * x**2 - 3))

    @staticmethod
    def _stoner_mass(n: Quantity, mu: float = REFERENCE_MU):
        """Re-implementation of equation (18a) in Stoner (1930), in terms of all the
        relevant quantities (i.e., with no pre-calculated numerical factors).

        This returns the total mass of the WD for a given number density at the
        equilibrium, and is used to calculate numerically the equilibrium radius.

        Arguments
        ---------
        n : Quantity (number density)
            The electron number density.

        mu : float
            The average number of nucleons per electron.
        """
        x = DegenerateFermiGas.fermi_momentum(n) / const.m_e / const.c
        return ((5. * const.h * const.c / 2**(1. / 3.) / const.G / (mu * const.m_p)**(4. / 3.)) * \
               (3. / 4. / np.pi)**(2. / 3.) * UniformWhiteDwarf._F(x))**(3. / 2.)

    @staticmethod
    def stoner_radius(M: Quantity, mu: float = REFERENCE_MU) -> Quantity:
        """Return the equilibrium radius for a uniform sphere derived in Stoner (1930)
        for the general case.

        Note this cannot be calculated analytically, and we resort to a spline over
        a precalculated grid of masses and radii.

        Arguments
        ---------
        M : Quantity (mass)
            The total mass of the star.

        mu : float
            The average number of nucleons per electron.
        """
        # Create a reasonable grid in the number density space.
        n = np.logspace(25., 45., 200) * u.cm**-3
        # Calculate the total mass of the star at all number densities on the grid.
        mass = UniformWhiteDwarf._stoner_mass(n).to(u.kg)
        # Convert the mass to a radius assuming a uniform sphere.
        radius = (3. * mass / 4. / np.pi / n / mu / const.m_p)**(1. / 3.)
        return WhiteDwarfBase.interpolate_radius(mass, radius, M)

    @staticmethod
    def stoner_limit(mu: float = REFERENCE_MU) -> Quantity:
        """Return the Stoner limit for the mass of a white dwarf.

        Arguments
        ---------
        mu : float
            The average number of nucleons per electron.
        """
        value = 3. / 4. / np.pi * (5. / 4.)**(3. / 2.) / np.sqrt(2) * \
                (const.h * const.c / const.G)**(3. / 2.) / mu**2. / const.m_p**2
        return value.to(u.kg)


class WhiteDwarf:

    """Class representing a white dwarf.

    This is based on S. Chandrasekhar, "The highly collapsed configurations of a
    stellar mass (Second paper)", Monthly Notices of the Royal Astronomical Society,
    Vol. 95, p.207-225.
    """

    REFERENCE_MU = WhiteDwarfBase.REFERENCE_MU

    # Table III in https://articles.adsabs.harvard.edu/pdf/1935MNRAS..95..207C
    # These are calculated for mu = 1. For a generic mu, masses should be divided
    # by mu**2, and radii by mu.
    _NORMALIZED_M = np.array([
        0.877, 1.612, 2.007, 2.440, 2.934, 3.528, 4.310, 4.852, 5.294, 5.484, 5.728
        ]) * const.M_sun
    _NORMALIZED_R = np.array([
        2.792, 2.155, 1.929, 1.721, 1.514, 1.287, 0.9936, 0.7699, 0.5443, 0.4136, 0.
        ]) * 1.e9 * u.cm

    @staticmethod
    def chandrasekhar_radius(M: Quantity, mu: float = REFERENCE_MU) -> Quantity:
        """Return the Chandrasekar radius for a given stellar mass.

        Arguments
        ---------
        M : Quantity (mass)
            The total mass of the star.

        mu : float
            The average number of nucleons per electron.
        """
        mass = WhiteDwarf._NORMALIZED_M / mu**2
        radius = WhiteDwarf._NORMALIZED_R / mu
        return WhiteDwarfBase.interpolate_radius(mass, radius, M)

    @staticmethod
    def chandrasekhar_limit(mu: float = REFERENCE_MU) -> Quantity:
        """Return the Chandrasekar limit for the mass of a white dwarf.

        Arguments
        ---------
        mu : float
            The average number of nucleons per electron.
        """
        omega_0_3 = 2.018236
        value = omega_0_3 * np.sqrt(3. * np.pi) / 2. * \
                (const.h * const.c / const.G / 2. / np.pi)**(3. / 2.) / mu**2. / const.m_p**2
        return value.to(u.kg)
