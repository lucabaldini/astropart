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

"""Facilities related to the CMB.
"""

from functools import partial

import numpy as np

from astropart import bb


_CMB_TEMPERATURE = 2.72548

spectral_energy_density_nu = partial(bb.spectral_energy_density_nu, temperature=_CMB_TEMPERATURE)

differential_number_density = partial(bb.differential_number_density, temperature=_CMB_TEMPERATURE)
number_density_peak = partial(bb.number_density_peak, temperature=_CMB_TEMPERATURE)
differential_energy_density = partial(bb.differential_energy_density, temperature=_CMB_TEMPERATURE)
energy_density_peak = partial(bb.energy_density_peak, temperature=_CMB_TEMPERATURE)


def integral_number_density(energy_min: float = 0., energy_max: float = np.inf) -> float:
    """Return the CMB number density in a given energy interval.

    Note that the default maximum energy is 0.1 eV in order to avoid a RuntimeWarning
    due to an overflow in the np.exp function when trying to calculate the
    differential spectrum at too high of an energy.

    Parameters
    ----------
    energy_min : float
        The minimum integration energy in eV.

    energy_max : float
        The maximum integration energy in eV.

    Returns
    -------
    float
        The CMB number density in the given energy range in cm^{-3}.
    """
    # Guard to avoid extending the numerical integral where the pdf is too small.
    energy_max = min(energy_max, 0.1)
    return bb.number_density(energy_min, energy_max, _CMB_TEMPERATURE, epsabs=1.e-20)


def integral_energy_density(energy_min: float = 0., energy_max: float = np.inf) -> float:
    """Return the CMB energy density in a given energy interval.

    Note that the default maximum energy is 0.1 eV in order to avoid a RuntimeWarning
    due to an overflow in the np.exp function when trying to calculate the
    differential spectrum at too high of an energy.

    Parameters
    ----------
    emin : float
        The minimum integration energy in eV.

    emax : float
        The maximum integration energy in eV.

    Returns
    -------
    float
        The CMB energy density in the given energy range in eV cm^{-3}.
    """
    # Guard to avoid extending the numerical integral where the pdf is too small.
    energy_max = min(energy_max, 0.1)
    return bb.energy_density(energy_min, energy_max, _CMB_TEMPERATURE, epsabs=1.e-20)
