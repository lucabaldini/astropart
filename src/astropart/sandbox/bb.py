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

"""Black-body radiation.
"""

from astropy import constants as const
from astropy import units as u
import numpy as np

from astropart.numeric import integral


# pylint: disable=no-member
_NORM_NU = 8. * np.pi * const.h.cgs.value / (const.c.cgs.value)**3.
_PEAK_SCALE_NU = 2.821439372122078893
_NORM_LAMBDA = 8. * np.pi * const.h.cgs.value * const.c.cgs.value
_PEAK_SCALE_LAMBDA = 4.965114231744276303
_NORM_ENERGY = 8. * np.pi / (const.h.to_value(u.eV * u.s) * const.c.cgs.value)**3.
_PEAK_SCALE_ENERGY = _PEAK_SCALE_NU
_PEAK_SCALE_NUMBER = 1.5936


def _planck_factor(x: float) -> float:
    """Convenience function calculating the Planck factor 1. / (exp(u) - 1.).
    """
    # pylint: disable=invalid-name
    return 1. / (np.exp(x) - 1.)


def spectral_energy_density_nu(nu: float, temperature: float) -> float:
    """Return the spectral energy density in frequency space in erg Hz^{-1} cm^{-3}.

    Parameters
    ----------
    nu : float
        The input frequency in Hz.

    temperature : float
        The blackbody temperature in deg K.

    Returns
    -------
    float
        The BB spectral energy density at a given frequency in erg Hz^{-1} cm^{-3}.
    """
    # pylint: disable=invalid-name
    x = const.h.cgs.value * nu / const.k_B.cgs.value / temperature
    return _NORM_NU * nu**3. * _planck_factor(x)


def integral_energy_density_nu(nu_min: float, nu_max: float, temperature: float, **kwargs) -> float:
    """Return the integral energy density within a given frequency range in erg cm^{-3}.

    Parameters
    ----------
    nu_min : float
        The minimum frequency in Hz.

    nu_max : float
        The maximum frequency in Hz.

    temperature : float
        The blackbody temperature in deg K.

    kwargs : dict
        Keyword arguments to be passed to scipy.quad for the underlying numeric integral.

    Returns
    -------
    float
        The BB energy density within the band in erg cm^{-3}.
    """
    return integral(spectral_energy_density_nu, nu_min, nu_max, (temperature,), **kwargs)


def spectral_peak_nu(temperature: float) -> float:
    """Return the position of the maximum of the energy density in frequency space.

    This is performing by numerically solving the underlying trascendental equation.

    Parameters
    ----------
    temperature : float
        The blackbody temperature in deg K.

    Returns
    -------
    float
        The position of the maximum of the energy density in Hz.
    """
    return _PEAK_SCALE_NU * const.k_B.cgs.value * temperature / const.h.cgs.value


def spectral_energy_density_lambda(lambda_: float, temperature: float) -> float:
    """Return the spectral energy density in wavelenght space in erg cm^{-4}.

    Parameters
    ----------
    lambda_ : float
        The input wavelength in cm.

    temperature : float
        The blackbody temperature in deg K.

    Returns
    -------
    float
        The BB differential energy density at a given wavelength in erg cm^{-4}.
    """
    # pylint: disable=invalid-name
    x = const.h.cgs.value * const.c.cgs.value / lambda_ / const.k_B.cgs.value / temperature
    return _NORM_LAMBDA / lambda_**5. * _planck_factor(x)


def integral_energy_density_lambda(lambda_min: float, lambda_max: float, temperature: float,
    **kwargs) -> float:
    """Return the overall energy density within a given wavelenght range.

    Parameters
    ----------
    lambda_min : float
        The minimum frequency in cm.

    lambda_max : float
        The maximum frequency in cm.

    temperature : float
        The blackbody temperature in deg K.

    kwargs : dict
        Keyword arguments to be passed to scipy.quad for the underlying numeric integral.

    Returns
    -------
    float
        The BB energy density within the band in erg cm^{-3}.
    """
    return integral(spectral_energy_density_lambda, lambda_min, lambda_max, (temperature,), **kwargs)


def spectral_peak_lambda(temperature: float) -> float:
    """Return the position of the maximum of the energy density in wavelenght space.

    This is performing by numerically solving the underlying trascendental equation.

    Parameters
    ----------
    temperature : float
        The blackbody temperature in deg K.

    Returns
    -------
    float
        The position of the maximum of the energy density in cm.
    """
    return const.h.cgs.value * const.c.cgs.value / const.k_B.cgs.value /\
        temperature / _PEAK_SCALE_LAMBDA


def differential_number_density(energy: float, temperature: float) -> float:
    """Return the differential number density in eV^{-1} cm^{-3}.

    Parameters
    ----------
    energy : float
        The input energy in eV.

    temperature : float
        The blackbody temperature in deg K.

    Returns
    -------
    float
        The BB differential number density at a given energy in eV^{-1} cm^{-3}.
    """
    # pylint: disable=invalid-name
    x = energy / const.k_B.to_value(u.eV / u.K) / temperature
    return _NORM_ENERGY * energy**2. * _planck_factor(x)


def integral_number_density(energy_min: float, energy_max: float, temperature: float,
    **kwargs) -> float:
    """Return the integral number density within a given energy range.

    Parameters
    ----------
    energy_min : float
        The minimum energy in eV.

    energy_max : float
        The maximum energy in eV.

    temperature : float
        The blackbody temperature in deg K.

    kwargs : dict
        Keyword arguments to be passed to scipy.quad for the underlying numeric integral.

    Returns
    -------
    float
        The BB number density within the band in cm^{-3}.
    """
    return integral(differential_number_density, energy_min, energy_max,
        (temperature,), **kwargs)


def number_density_peak(temperature: float) -> float:
    """Return the position of the maximum of the number density in eV.

    This is performing by numerically solving the underlying trascendental equation.

    Parameters
    ----------
    temperature : float
        The blackbody temperature in deg K.

    Returns
    -------
    float
        The position of the maximum of the number density in eV.
    """
    return _PEAK_SCALE_NUMBER * const.k_B.to_value(u.eV / u.K) * temperature


def differential_energy_density(energy: float, temperature: float) -> float:
    """Return the differential energy density in energy space.

    Parameters
    ----------
    energy : float
        The input energy in eV.

    temperature : float
        The blackbody temperature in deg K.

    Returns
    -------
    float
        The BB differential energy density at a given energy in 1. / cm^3.
    """
    return energy * differential_number_density(energy, temperature)


def integral_energy_density(energy_min: float, energy_max: float, temperature: float,
    **kwargs) -> float:
    """Return the overall energy density within a given energy range in eV cm^{-3}.

    Parameters
    ----------
    energy_min : float
        The minimum energy in eV.

    energy_max : float
        The maximum energy in eV.

    temperature : float
        The blackbody temperature in deg K.

    kwargs : dict
        Keyword arguments to be passed to scipy.quad for the underlying numeric integral.

    Returns
    -------
    float
        The BB energy density within the band in eV cm^{-3}.
    """
    return integral(differential_energy_density, energy_min, energy_max,
        (temperature,), **kwargs)


def energy_density_peak(temperature: float) -> float:
    """Return the position of the maximum of the energy density in eV.

    This is performing by numerically solving the underlying trascendental equation.

    Parameters
    ----------
    temperature : float
        The blackbody temperature in deg K.

    Returns
    -------
    float
        The position of the maximum of the energy density in eV.
    """
    return _PEAK_SCALE_ENERGY * const.k_B.to_value(u.eV / u.K) * temperature
