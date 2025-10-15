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

"""Neutron stars and pulsars.
"""

from astropy import constants as const
from astropy import units as u
from astropy.units.quantity import Quantity
import numpy as np

from astropart import ASTROPART_DATA
from astropart.plot import plt, setup_ca


class PulsarCatalog:

    """Compact interface to the ATNF pulsar catalog.

    When you use this, acknowledge use of the ATNF Pulsar Catalogue, both by giving
    the Catalogue web address (https://www.atnf.csiro.au/research/pulsar/psrcat/)
    and by referencing the paper:
    Manchester, R. N., Hobbs, G.B., Teoh, A. & Hobbs, M., AJ, 129, 1993-2006 (2005)
    """

    PSRCAT_FILE_PATH = ASTROPART_DATA / 'atnf_psrcat.csv'
    COL_DEF = [
        ('row', 'i8'), # Row number, e.g., 274
        ('name', 'U50'), # Pulsar name, e.g., B0531+21
        ('psrj', 'U50'), # Pulsar name based on J2000 coordinates, e.g., J0534+2200
        ('raj', 'U20'), # Right ascension, e.g., 05:34:31.9
        ('decj', 'U20'), # Declination, e.g., +22:00:52.1
        ('period', 'f4'), # Period, e.g., 0.033392 [s]
        ('pdot', 'f4'), # Period derivative, e.g., 4.21e-13
        ('dm', 'f4'), # Dispersion measure, e.g., 56.77 [pc cm^-3]
        ('distance', 'f4'), # Distance, e.g., 2.0 [kpc]
        ('associations', 'U100'), # Names of other objects associated with the pulsar
        ('age', 'f4'), # Age, e.g., 1.26e+03 [yr]
        ('bsurf', 'f4'), # Characteristic surface magnetic field, e.g., 3.79e+12 [G]
        ('edot', 'f4') # Spin-down luminosity, e.g., 4.46e+38 [erg s^-1]
        ]

    def __init__(self):
        """Constructor.

        This is reading in the ATNF pulsar catalog.
        """
        kwargs = dict(delimiter=';', dtype=self.COL_DEF, unpack=True, encoding='utf-8',
                      missing_values='*')
        _, self.name, _, _, _, self.period, self.pdot, _, self.distance, self.associations, \
            self.age, self.bsurf, self.edot = np.genfromtxt(self.PSRCAT_FILE_PATH, **kwargs)

    def plot_ppdot(self, max_distance: float = 2., marker='o', **kwargs):
        """Plot the catalogued pulsars onto a P-Pdot diagram.
        """
        mask = self.distance <= max_distance
        plt.plot(self.period[mask], self.pdot[mask], marker, **kwargs)

    def find_row(self, pattern: str) -> int:
        """Find a pulsar by associations.

        Arguments
        ---------
        pattern : str
            The pattern to search for.
        """
        for row, association in enumerate(self.associations):
            if pattern in association.lower():
                return row
        return None


class CanonicalNeutronStar:

    """Definition of the canonical neutron star, that is, a sphere with 1.4 solar
    masses in a 10 km radius.
    """

    MASS = 1.4 * const.M_sun
    RADIUS = 10. * u.km
    I = (2. / 5. * MASS * RADIUS**2.).to(u.g * u.cm**2.)


class CanonicalPulsar(CanonicalNeutronStar):

    """Canonical pulsar.
    """

    @staticmethod
    def minimum_density(P: Quantity) -> Quantity:
        """Return the minimum density derived from the condition that the equatorial
        centrifugal force does not exceed the gravitational attraction.

        Arguments
        ---------
        P : Quantity (time)
            The pulsar period.
        """
        value = 3. * np.pi / const.G / P**2.
        return value.to(u.g / u.cm**3)

    @classmethod
    def rotational_energy(cls, P: Quantity) -> Quantity:
        """Return the rotational kinetic energy.

        Arguments
        ---------
        P : Quantity (time)
            The pulsar period.
        """
        value = 2 * np.pi**2 * cls.I / P**2.
        return value.to(u.erg)

    @classmethod
    def characteristic_magnetic_field_prefactor(cls) -> Quantity:
        """Return the characteristic magnetic field prefactor.

        Note we have to use brute force to circumvent the differences between MKS and cgs.
        See https://github.com/astropy/astropy/issues/4359 for more details.
        """
        prefactor = (3. * const.c**3. * cls.I / 8. / np.pi**2. / cls.RADIUS**6.)**0.5
        return prefactor.cgs.value * u.G / u.s**0.5

    @classmethod
    def characteristic_magnetic_field(cls, P: Quantity, pdot: float) -> Quantity:
        """Return the characteristic magnetic field.

        Arguments
        ---------
        P : Quantity (time)
            The pulsar period.

        pdot : float
            The pulsar period derivative.
        """
        value = cls.characteristic_magnetic_field_prefactor() * (P * pdot)**0.5
        return value.to(u.G)

    @staticmethod
    def characteristic_age(P: Quantity, pdot: float) -> Quantity:
        """Return the characteristic age.

        Arguments
        ---------
        P : Quantity (time)
            The pulsar period.

        pdot : float
            The pulsar period derivative.
        """
        age = P / 2. / pdot
        return age.to(u.year)

    @classmethod
    def ppdot_figure(cls):
        """Create the standard pusar p-pdot diagram.
        """
        P = np.geomspace(1.e-3, 1.e2) * u.s
        C = cls.characteristic_magnetic_field_prefactor()
        for B in np.geomspace(1.e8, 1.e14, 4) * u.G:
            pdot = (B / C)**2. / P
            plt.plot(P, pdot, color='black')
            x = 2.e-3 * u.s
            y = (B / C)**2. / x
            plt.text(x.value, 0.4 * y.value, f'$10^{{{np.log10(B.value):.0f}}}$ G', rotation=-12., va='top')
        for age in np.geomspace(1.e3, 1.e9, 3) * u.year:
            pdot = P.to(u.year) / 2. / age
            plt.plot(P, pdot, color='black')
            x = 20 * u.s
            y = x.to(u.year) / 2. / age
            plt.text(x.value, 1.5 * y.value, f'$10^{np.log10(age.value):.0f}$ yr', rotation=20)
        setup_ca(xlabel='$P$ [s]', ylabel='$\\dot{P}$', logx=True, logy=True,
                ymin=1.e-22, ymax=1.e-8)
