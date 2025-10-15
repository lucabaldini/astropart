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

"""Standard solar model.
"""

import numpy as np

from astropart import ASTROPART_DATA, logger
from astropart.numeric import Spline


class _BS05:

    """Solar model described in https://arxiv.org/pdf/astro-ph/0412440.pdf

    Columns in the Standard Model table represent:

    1)  Mass fraction in units of the solar mass
    2)  Radius of the zone in units of one solar radius
    3)  Temperature in units of deg (K)
    4)  Density in units of g/cm^3
    5)  Pressure in units of dyn/cm^2
    6)  Luminosity fraction in units of the solar luminosity
    7)  X(^1H): the hydrogen mass fraction
    8)  X(^4He): the helium 4 mass fraction
    9)  X(^3He): the helium 3 mass fraction
    10) X(^12C): the carbon 12 mass fraction
    11) X(^14N): the nitrogen 14 mass fraction
    12) X(^16O): the oxygen 16 mass fraction

    Columns in the neutrino flux table represent:

    1) Radius of the zone in units of one solar radius
    2) Temperature in units of 10^6 deg (K)
    3) Logarithm (to the base 10) of the electron density in units of
       cm^{-3}/N_A, where N_A is the Avogadro number
    4) Mass of the zone with the given radius in units of one solar mass
    5) X(^7Be): beryllium-7 mass fraction
       [For special purposes, it is useful to know the 7^Be mass fraction.]
    6) Fraction of pp neutrinos produced in the zone
    7) Fraction of boron 8 neutrinos produced in the zone
    8) Fraction of nitrogen 13 neutrinos produced in the zone
    9) Fraction of oxygen 15 neutrinos produced in the zone
    10) Fraction of florine 17 neutrinos produced in the zone
    11) Fraction of beryllium 7 neutrinos produced in the zone
    12) Fraction of pep neutrinos produced in the zone
    13) Fraction of hep neutrinos produced in the zone
    """

    _BS05_FILE_PATH = ASTROPART_DATA / 'bs05_agsop.dat'
    _BS05_NUFLUX_FILE_PATH = ASTROPART_DATA / 'bs2005agsopflux.dat'

    def __init__(self):
        """Constructor.
        """
        logger.info(f'Loading SSM B05 data from {self._BS05_FILE_PATH}...')
        m, r, T, rho, P, L, x1, x4, x3, x12, x14, x16 = np.loadtxt(self._BS05_FILE_PATH, unpack=True)
        self.cumulative_mass = Spline(r, m, ext=3)
        self.temperature = Spline(r, T, ext=3)
        self.density = Spline(r, rho, ext=3)
        self.pressure = Spline(r, P, ext=3)
        self.luminosity = Spline(r, L, ext=3)
        self.mass_fraction_1H = Spline(r, x1, ext=3)
        self.mass_fraction_4He = Spline(r, x4, ext=3)
        self.mass_fraction_3He = Spline(r, x3, ext=3)
        self.mass_fraction_12C = Spline(r, x12, ext=3)
        self.mass_fraction_14N = Spline(r, x14, ext=3)
        self.mass_fraction_16O = Spline(r, x16, ext=3)
        logger.info(f'Loading SSM B05 neutrino flux data from {self._BS05_NUFLUX_FILE_PATH}...')
        r, T, ne, m, _, xpp, xB8, xN13, xO15, xF17, xBe7, xpep, xhep = np.loadtxt(self._BS05_NUFLUX_FILE_PATH, unpack=True)
        self.electron_numer_density = Spline(r, 10.**ne * 6.02e23)
        self.pdf_flux_pp = Spline(r, xpp)
        self.pdf_flux_B8 = Spline(r, xB8)
        self.pdf_flux_N13 = Spline(r, xN13)
        self.pdf_flux_015 = Spline(r, xO15)
        self.pdf_flux_F17 = Spline(r, xF17)
        self.pdf_flux_Be7 = Spline(r, xBe7)
        self.pdf_flux_pep = Spline(r, xpep)
        self.pdf_flux_hep = Spline(r, xhep)


BS05 = _BS05()
