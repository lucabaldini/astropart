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

"""Some useful physical quantities.
"""


import astropy.constants as const
import astropy.units as u


class Quantity(u.Quantity):

    """This is a sligthtly customized subclass of astropy.units.Quantity.

    While a quantity holds a value and a unit, here we are adding two additional
    fields, name and reference, modeled on astropy.constants.Constant.

    See https://docs.astropy.org/en/stable/units/quantity.html for some notes on
    subclassing astropy.units.Quantity

    Note that we provide classmethods to create Quantity objects from either
    astropy.units.Quantity or astropy.constants.Constant instances.
    """

    def __new__(cls, name : str, value : float, unit : u.core.Unit, reference : str = None):
        """Create a new Quantity instance.
        """
        instance = u.Quantity.__new__(cls, value, unit)
        instance.name = name
        instance.reference = reference
        return instance

    @classmethod
    def from_quantity(cls, name : str, quantity : u.Quantity):
        """Create a class instance from an astropy quantity.
        """
        return cls(name, quantity.value, quantity.unit)

    @classmethod
    def from_unit(cls, name, unit : u.core.Unit):
        """Create a class instance from an astropy unit.
        """
        return cls.from_quantity(name, 1. * unit)

    @classmethod
    def from_constant(cls, const : const.Constant):
        """Create a class instance from an astropy Constant.
        """
        return cls(const.name, const.value, const.unit)

    def __str__(self):
        """String formatting.
        """
        return f'{self.name}: {u.Quantity.__str__(self)}'



class Range:

    """Small convenience class expressing a range of values (e.g., for the mass
    or radius of a white dwarf).
    """

    def __init__(self, name : str, minimum : float, maximum : float, unit, reference : str = None):
        """Constructor.
        """
        self.minimum = minimum * unit
        self.maximum = maximum * unit
        self.name = name
        self.unit = unit
        self.reference = reference

    def __str__(self):
        """String formatting.
        """
        return f'{self.minimum.value}--{self.maximum.value} {self.unit.name}'



# Spatial dimensions.
PROTON_RADIUS = Quantity('Proton radius', 0.84184e-15, u.m, 'https://ui.adsabs.harvard.edu/abs/2010Natur.466..213P')
BOHR_RADIUS = Quantity.from_constant(const.a0)
NEUTRON_STAR_RADIUS = Quantity('1.4 solar-masses neutron star radius', 11., u.km, 'https://ui.adsabs.harvard.edu/abs/2020NatAs...4..625C')
EARTH_RADIUS = Quantity.from_constant(const.R_earth)
WHITE_DWARF_RADIUS = Range('White dwarf radius', 0.008, 0.02, u.R_earth)
JUPITER_RADIUS = Quantity.from_constant(const.R_jup)
SUN_RADIUS = Quantity.from_constant(const.R_sun)
RED_GIANT_RADIUS = Range('Red giant radius', 100., 1000., u.R_sun)
NEPTUNE_APHELION = Quantity('Neptune aphelion', 30.33, u.au, 'https://en.wikipedia.org/wiki/Neptune')
OMEGA_CENTAURI_DIAMETER = Quantity('Omega Centauri diameter', 2. * 86., u.lyr, 'https://en.wikipedia.org/wiki/Omega_Centauri')
MILKY_WAY_RADIUS = Quantity('Milky way diameter', 13.4, u.kpc, 'https://en.wikipedia.org/wiki/Milky_Way')
VIRGO_RADIUS = Quantity('Virgo cluster diameter', 2.2, u.Mpc, 'https://en.wikipedia.org/wiki/Virgo_Cluster')
UNIVERSE_RADIUS = Quantity('Diameter of the observable universe', 28.5, u.Gpc, 'https://en.wikipedia.org/wiki/Observable_universe')

# Distances
DIST_TO_MOON = Quantity('Average Moon distance to Earth', 384399., u.km, 'https://en.wikipedia.org/wiki/Lunar_distance_(astronomy)')
DIST_TO_SUN = Quantity.from_unit('Average Sun distance to Earth', u.au)
DIST_TO_PROXIMA_CEN = Quantity('Distance to Proxima Centauri', 4.2465, u.lyr, 'https://en.wikipedia.org/wiki/Proxima_Centauri')
DIST_TO_SIRUS_B = Quantity('Distance to Sirius B', 8.6, u.lyr, 'https://en.wikipedia.org/wiki/White_dwarf')
DIST_TO_CRAB = Quantity('Distance to Crab Nebula', 6.5, u.klyr, 'https://en.wikipedia.org/wiki/Crab_Nebula')
DIST_TO_GALACTIC_CENTER = Quantity('Distance to Galactic Center', 26., u.klyr, 'https://en.wikipedia.org/wiki/Galactic_Center')
DIST_TO_LMC = Quantity('Distance to LMC', 49.97, u.kpc, 'https://en.wikipedia.org/wiki/Large_Magellanic_Cloud')
DIST_TO_VIRGO = Quantity('Distance to Virgo Cluster', 16.5, u.Mpc, 'https://en.wikipedia.org/wiki/Virgo_Cluster')
DIST_TO_BL_LAC = Quantity('Distance to BL Lacertae', 276., u.Mpc, 'https://en.wikipedia.org/wiki/BL_Lacertae')
DIST_TO_HD1 = Quantity('Proper distance to HD1', 10.206, u.Gpc, 'https://en.wikipedia.org/wiki/HD1_(galaxy)')

# Masses
ELECTRON_MASS = Quantity.from_constant(const.m_e)
PROTON_MASS = Quantity.from_constant(const.m_p)
EARTH_MASS = Quantity.from_constant(const.M_earth)
JUPITER_MASS = Quantity.from_constant(const.M_jup)
WHITE_DWARF_MASS = Range('White dwarf mass', 0.17, 1.33, u.M_sun)
SUN_MASS = Quantity.from_constant(const.M_sun)
MILKY_WAY_MASS = Quantity('Mass of the Milky Way', 1.15e12, u.M_sun)
UNIVERSE_MASS = Quantity('Mass of the observable universe', 1.5e53, u.kg, 'https://en.wikipedia.org/wiki/Observable_universe')

# Energies


# Temperatures


if __name__ == '__main__':
    for item in dir():
        if item.startswith('__'):
            continue
        if item in ('const', 'u', 'Quantity', 'Range'):
            continue
        print(f'{item} -> {eval(item)}')
    print(f'1 pc = {u.pc.to(u.au)} au = {u.pc.to(u.m)}')
    print(f'0.3 as = {u.arcsec.to(u.rad, 0.3)} rad')
