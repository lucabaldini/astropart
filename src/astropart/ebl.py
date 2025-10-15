# Copyright (C) 2024-2025, Luca Baldini (luca.baldini@pi.infn.it)
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

"""EBL distribution.
"""

import astropy.constants as const
import astropy.units as u
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from astropart import ASTROPART_DATA, logger


_EBL_SED_FILE_PATH = ASTROPART_DATA / 'ebl_universe_sed.csv'


def _build_spline(x: np.ndarray, y: np.ndarray, k: int = 3) -> InterpolatedUnivariateSpline:
    """Convenience method to build an interpolated spline.
    """
    return InterpolatedUnivariateSpline(x, y, k=k)


def _load_spectral_density(file_path: str = _EBL_SED_FILE_PATH) -> tuple:
    """Load a SED from a given file path.

    It is assumed that the file contains exactly two columns, representing:

    * the frequency of the radiation in GHz;
    * the frequency brightness nu I_nu in nW m^{-2} sr^{-1}.

    Parameters
    ----------
    file_path : str
        The path to the file containing the input data.

    Returns
    -------
    tuple
        A two-element tuple containing the frequency and corresponding spectral
        density in cgs units.
    """
    logger.info(f'Loading EBL SED data from {file_path}...')
    nu, brightness = np.loadtxt(file_path, delimiter=',', unpack=True)
    # The frequency is in GHz.
    nu = nu * u.GHz
    # And the y column is the brightness times the frequency, so we need to divide
    # by the frequency (in Hz) in order to get the actual brightness.
    brightness /= nu.to_value(u.Hz)
    brightness *= u.nW / u.m**2. / u.Hz / u.sr
    spectral_density = 4. * np.pi * u.sr / const.c * brightness
    return nu.to_value(u.Hz), spectral_density.to_value(u.erg / u.Hz / u.cm**3.)


_EBL_SPEC_DENSITY_DATA = _load_spectral_density()


def _energy_density_data(spec_data=_EBL_SPEC_DENSITY_DATA):
    """Convert the spectral energy density data into the corresponding energy
    representation (i.e., the frequency into the corresponding energy in eV and
    the spectral energy density into an energy density in cm^{-3}).
    """
    nu, spectral_density = spec_data
    energy = const.h.to_value(u.eV * u.s) * nu
    energy_density = spectral_density / const.h.to_value(u.erg * u.s)
    return energy, energy_density


def _build_spectral_energy_density_spline(spec_data=_EBL_SPEC_DENSITY_DATA,
    k: int = 2) -> InterpolatedUnivariateSpline:
    """Build a spline with the EBL spectral energy density.
    """
    return _build_spline(*spec_data, k)


def _build_differential_energy_density_spline(spec_data=_EBL_SPEC_DENSITY_DATA,
    k: int = 2) -> InterpolatedUnivariateSpline:
    """Build a spline with the EBL differential energy density.
    """
    return _build_spline(*_energy_density_data(spec_data), k)


def _build_differential_number_density_spline(spec_data=_EBL_SPEC_DENSITY_DATA,
    k: int = 2) -> InterpolatedUnivariateSpline:
    """Build a spline with the EBL differential number density.
    """
    energy, energy_density = _energy_density_data(spec_data)
    return _build_spline(energy, energy_density / energy, k)


# These are the main interaces, and are modeled on those in the cmb module.
spectral_energy_density_nu = _build_spectral_energy_density_spline()
differential_energy_density = _build_differential_energy_density_spline()
differential_number_density =  _build_differential_number_density_spline()


def integral_energy_density(energy_min: float, energy_max: float) -> float:
    """Return the integral energy density within a given energy band.
    """
    return differential_energy_density.integral(energy_min, energy_max)


def integral_number_density(energy_min: float, energy_max: float) -> float:
    """Return the integral number density within a given energy band.
    """
    return differential_number_density.integral(energy_min, energy_max)
