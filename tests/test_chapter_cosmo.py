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

"""Proofreading for the cosmology chapter.
"""

from loguru import logger

import astropy.constants as const
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import numpy as np

from astropart.cosmo import geff_energy, temperature_to_cosmic_time


def test_chapter():

    for kT in (150, 10, 0.8, 0.5, 0.1, 0.25e-6):
        kT *= u.MeV
        geff = geff_energy(kT)
        t = temperature_to_cosmic_time(kT, geff)
        print(f'Time when T was {kT} (geff = {geff:.2f}) = {t.to(u.s):.3e}')
