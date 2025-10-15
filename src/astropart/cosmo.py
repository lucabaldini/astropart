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

"""Cosmological quantities.
"""


import astropy.constants as const
import astropy.units as u
from astropy.units.quantity import Quantity
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from astropart import logger, ASTROPART_DATA


_GEFF_FILE_PATH = ASTROPART_DATA / 'geff_standard_2018.dat'

M_P = np.sqrt(const.hbar * const.c / const.G)
_TEMP_CONST = M_P * const.c**2 / 2 * np.sqrt(45. * const.h**2 / 16. / np.pi**5)



def _load_geff_data(file_path: str = _GEFF_FILE_PATH):
    """Load the effective degrees of freedom from a given file path.
    """
    logger.info(f'Loading geff data from {file_path}...')
    kT, geff, _, geffs, _ = np.loadtxt(file_path, unpack=True)
    # Convert the temperature from GeV to MeV.
    kT *= 1.e3
    # Crete the splines.
    geff_spline = InterpolatedUnivariateSpline(kT, geff)
    geffs_spline = InterpolatedUnivariateSpline(kT, geffs)
    return geff_spline, geffs_spline


_GEFF_SPLINE, _GEFFS_SPLINE = _load_geff_data()


def geff_energy(kT: Quantity) -> float:
    """Return the effective relativistic degrees of freedom for the energy at a
    given temperature kT.
    """
    kT = kT.to(u.MeV)
    return _GEFF_SPLINE(kT)


def geff_entropy(kT: Quantity) -> float:
    """Return the effective relativistic degrees of freedom for the entropy at a
    given temperature kT.
    """
    kT = kT.to(u.MeV)
    return _GEFFS_SPLINE(kT)


def temperature_to_cosmic_time(kT: Quantity, geff: float = None) -> Quantity:
    """Return the cosmic time at which a radiation-dominated universe has a temperature
    corresponding to a given energy kT.
    """
    if geff is None:
        geff = geff_energy(kT)
    time = _TEMP_CONST / np.sqrt(geff) / kT**2
    return time.to(u.s)


# Create a spline for the temperature as a function of the cosmic time.
# Note this is using MeV and s internally.
_kT = np.geomspace(1.e6, 1.e-6, 200) * u.MeV
_TIME_SPLINE = InterpolatedUnivariateSpline(temperature_to_cosmic_time(_kT).to(u.s), _kT)


def cosmic_time_to_temperature(t):
    """
    """
    t = t.to(u.s)
    return _TIME_SPLINE(t) * u.MeV
