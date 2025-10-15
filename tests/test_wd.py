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

"""Unit tests for astropart.wd
"""

import astropy.constants as const
import astropy.units as u
import numpy as np

from astropart.plot import plt, setup_ca
from astropart.wd import UniformWhiteDwarf, WhiteDwarf


def test_chandrasekhar():
    """
    """
    mass = np.linspace(0.25, 2.0, 200) * const.M_sun
    radius_chandrasekhar = WhiteDwarf.chandrasekhar_radius(mass)
    radius_stoner_nr = UniformWhiteDwarf.non_relativistic_stoner_radius(mass)
    radius_stoner = UniformWhiteDwarf.stoner_radius(mass)
    plt.plot(mass / const.M_sun, radius_chandrasekhar.to(u.km), label='Chandrasekar')
    plt.plot(mass / const.M_sun, radius_stoner_nr.to(u.km), label='Stoner (NR)')
    plt.plot(mass / const.M_sun, radius_stoner.to(u.km), label='Stoner')
    plt.axhline(const.R_earth.to(u.km).value, ls='dashed')
    setup_ca(xlabel='Stellar mass [$M_\\odot$]', ylabel='Radius [km]', legend=True)


if __name__ == '__main__':
    test_chandrasekhar()
    plt.show()
