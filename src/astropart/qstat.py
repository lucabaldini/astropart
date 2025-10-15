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

"""Quantum statistics.
"""

from functools import partial

from astropy import constants as const
from astropy import units as u
from astropy.units.quantity import Quantity
from astropy.units.core import Unit
import numpy as np
import scipy.special

from astropart.plot import plt, setup_ca


# This should be moved into a separate module.
u.eV_c = u.def_unit('eV / c', u.eV / const.c)
u.keV_c = u.def_unit('keV / c', u.keV / const.c)
u.MeV_c = u.def_unit('MeV / c', u.MeV / const.c)
u.GeV_c = u.def_unit('GeV / c', u.GeV / const.c)


class DistributionBase:

    """Base class collecting functions common to distribution.
    """

    def exp_E_KT(E: Quantity, T: Quantity) -> float:
        """Calculate the basic exponential term exp(E / KT) common to all distributions.

        Note internally this is converting E and KT to the same energy units, and
        returning a numpy array without dimensions.

        Arguments
        ---------
        E : Quantity (energy)
            Energy.

        T : Quantity (temperature)
            Temperature.
        """
        unit = u.eV
        return np.exp(E.to(unit).value / (const.k_B * T).to(unit).value)


class BoseEinstein:

    """Bose-Einstein distribution.
    """

    @staticmethod
    def dfunc(E: Quantity, T: Quantity, mu: Quantity) -> float:
        """Return the main part of the distribtion function (excluding g).

        Arguments
        ---------
        E : Quantity (energy)
            Energy.

        T : Quantity (temperature)
            Temperature.

        mu : Quantity (energy)
            Chemical potential.
        """
        return 1. / (DistributionBase.exp_E_KT(E - mu, T) - 1.)


class FermiDirac:

    """Fermi-Dirac distribution.
    """

    @staticmethod
    def dfunc(E: Quantity, T: Quantity, mu: Quantity) -> float:
        """Return the main part of the distribtion function (excluding g).

        Arguments
        ---------
        E : Quantity (energy)
            Energy.

        T : Quantity (temperature)
            Temperature.

        mu : Quantity (energy)
            Chemical potential.
        """
        return 1. / (DistributionBase.exp_E_KT(E - mu, T) + 1.)


class Planck:

    """Container class describing the Planck distribution.
    """

    def __init__(self, temperature: Quantity):
        """Constructor.

        This creates a "frozen" instance of the class, i.e., one where all the function
        calls are frozen to the temperature passed as an argument.
        """
        self.temperature = temperature
        self.spectral_energy_density = self._freeze(Planck.spectral_energy_density)
        self.peak_frequency = self._freeze(Planck.peak_frequency)
        self.peak_wavelength = self._freeze(Planck.peak_wavelength)
        self.peak_energy = self._freeze(Planck.peak_energy)
        self.average_energy = self._freeze(Planck.average_energy)
        self.radiant_emittance = self._freeze(Planck.radiant_emittance)
        self.energy_density = self._freeze(Planck.energy_density)
        self.pressure = self._freeze(Planck.pressure)
        self.spectral_number_density = self._freeze(Planck.spectral_number_density)
        self.number_density = self._freeze(Planck.number_density)
        # Note here we cannot use partial because T comes first in the function
        # definition.
        self.plot_spectral_energy_density = \
            lambda unit: Planck.plot_spectral_energy_density(self.temperature, unit)
        self.plot_spectral_number_density = \
            lambda unit: Planck.plot_spectral_number_density(self.temperature, unit)

    def _freeze(self, method):
        """Freeze a class staticmethod at the temperature of a given class instance.
        """
        return partial(method, T=self.temperature)

    @staticmethod
    def _du_frequency(nu: Quantity, T: Quantity) -> Quantity:
        """Return the spectral energy density in frequency space.

        Arguments
        ---------
        nu : Quantity (frequency)
            Frequency.

        T : Quantity (temperature)
            Temperature.
        """
        E = const.h * nu
        value = 8. * np.pi * const.h / const.c**3 * nu**3 * BoseEinstein.dfunc(E, T, 0.)
        return value.to(u.eV / u.cm**3 / nu.unit)

    @staticmethod
    def _du_wavelength(lambda_: Quantity, T: Quantity) -> Quantity:
        """Return the spectral energy density in wavelength space.

        Arguments
        ---------
        lambda_ : Quantity (length)
            Wavelength.

        T : Quantity (temperature)
            Temperature.
        """
        E = const.h * const.c / lambda_
        value = 8. * np.pi * const.h * const.c * lambda_**-5 * BoseEinstein.dfunc(E, T, 0.)
        return value.to(u.eV / u.cm**3 / lambda_.unit)

    @staticmethod
    def _du_energy(E: Quantity, T: Quantity) -> Quantity:
        """Return the spectral energy density in energy space.

        Arguments
        ---------
        E : Quantity (energy)
            Energy.

        T : Quantity (temperature)
            Temperature.
        """
        value = 8. * np.pi / (const.h * const.c)**3 * E**3 * BoseEinstein.dfunc(E, T, 0.)
        return value.to(u.eV / u.cm**3 / E.unit)

    @staticmethod
    def spectral_energy_density(x: Quantity, T: Quantity) -> Quantity:
        """Convenience method wrapping the actual implementation of the spectral
        energy density in frequency, wavelength and energy space.

        Arguments
        ---------
        x : Quantity (frequency, wavelength, or energy)
            The value(s) of the independent variable where the spectral density should
            be evaluated.

        T : Quantity (temperature)
            Temperature.
        """
        if x.unit.physical_type == 'frequency':
            return Planck._du_frequency(x, T)
        if x.unit.physical_type == 'length':
            return Planck._du_wavelength(x, T)
        if x.unit.physical_type == 'energy':
            return Planck._du_energy(x, T)
        raise ValueError(f'Wrong units ({unit}) passed to Planck.spectral_energy_density()')

    @staticmethod
    def peak_frequency(T: Quantity) -> Quantity:
        """Wien displacement law in frequency space.

        Arguments
        ---------
        T : Quantity (temperature)
            Temperature.
        """
        value = 2.82144 * const.k_B / const.h * T
        return value.to(u.Hz)

    @staticmethod
    def peak_wavelength(T: Quantity) -> Quantity:
        """Wien displacement law in wavelength space.

        Arguments
        ---------
        T : Quantity (temperature)
            Temperature.
        """
        value = const.h * const.c / const.k_B / 4.96511 / T
        return value.to(u.cm)

    @staticmethod
    def peak_energy(T: Quantity) -> Quantity:
        """Wien displacement law in energy space.

        Arguments
        ---------
        T : Quantity (temperature)
            Temperature.
        """
        value = const.h * Planck.peak_frequency(T)
        return value.to(u.eV)

    @staticmethod
    def average_energy(T: Quantity) -> Quantity:
        """Return the average photon energy.

        Arguments
        ---------
        T : Quantity (temperature)
            Temperature.
        """
        value = 2.7012 * const.k_B * T
        return value.to(u.eV)

    @staticmethod
    def radiant_emittance(T: Quantity) -> Quantity:
        """Stefan-Boltzmann law.

        Arguments
        ---------
        T : Quantity (temperature)
            Temperature.
        """
        value = const.sigma_sb * T**4
        return value.to(u.eV / u.cm**2 / u.s)

    @staticmethod
    def energy_density(T: Quantity) -> Quantity:
        """Return the overall energy density for the radiation.

        Arguments
        ---------
        T : Quantity (temperature)
            Temperature.
        """
        value = 4. / const.c * Planck.radiant_emittance(T)
        return value.to(u.eV / u.cm**3)

    @staticmethod
    def pressure(T: Quantity) -> Quantity:
        """Return the radiation pressure.

        Arguments
        ---------
        T : Quantity (temperature)
            Temperature.
        """
        value = Planck.energy_density(T) / 3.
        return value.to(u.bar)

    @staticmethod
    def _dn_energy(E: Quantity, T: Quantity) -> Quantity:
        """Return the spectral number density in energy space.

        Arguments
        ---------
        E : Quantity (energy)
            Energy.

        T : Quantity (temperature)
            Temperature.
        """
        value = 8. * np.pi / (const.h * const.c)**3 * E**2 * BoseEinstein.dfunc(E, T, 0.)
        return value.to(u.cm**-3 / u.eV)

    @staticmethod
    def spectral_number_density(x: Quantity, T: Quantity) -> Quantity:
        """Convenience method wrapping the actual implementation of the spectral
        number density in energy space.

        Arguments
        ---------
        x : Quantity (energy)
            The value(s) of the independent variable where the spectral density should
            be evaluated.

        T : Quantity (temperature)
            Temperature.
        """
        if x.unit.physical_type == 'energy':
            return Planck._dn_energy(x, T)
        raise ValueError(f'Wrong units ({unit}) passed to Planck.spectral_number_density()')

    @staticmethod
    def number_density(T: Quantity) -> Quantity:
        """Return the number density.

        Arguments
        ---------
        T : Quantity (temperature)
            Temperature.
        """
        value = 16. * np.pi * scipy.special.zeta(3) * (const.k_B * T / const.h / const.c)**3
        return value.to(u.cm**-3)

    @staticmethod
    def standard_grid(T: Quantity, unit: Unit = u.eV, num_points: int = 200) -> np.array:
        """Return a standard grid to plot the distribution.
        """
        if unit.physical_type == 'frequency':
            peak = Planck.peak_frequency(T).to(unit)
            return np.linspace(0.01 * peak, 5. * peak, num_points)
        if unit.physical_type == 'length':
            peak = Planck.peak_wavelength(T).to(unit)
            return np.linspace(0.2 * peak, 8. * peak, num_points)
        if unit.physical_type == 'energy':
            peak = Planck.peak_energy(T).to(unit)
            return np.linspace(0.01 * peak, 5. * peak, num_points)
        raise ValueError(f'Wrong units ({unit}) passed to Planck.standard_grid()')

    @staticmethod
    def plot_spectral_energy_density(T: Quantity, unit: Unit = u.eV) -> None:
        """Plot the spectral energy density.
        """
        x = Planck.standard_grid(T, unit)
        y = Planck.spectral_energy_density(x, T)
        plt.plot(x, y)
        if unit.physical_type == 'frequency':
            kwargs = dict(xlabel=f'Frequency [{unit}]')
        if unit.physical_type == 'length':
            kwargs = dict(xlabel=f'Wavelength [{unit}]')
        if unit.physical_type == 'energy':
            kwargs = dict(xlabel=f'Energy [{unit}]')
        setup_ca(ylabel=f'Spectral energy density [{y.unit.to_string("latex")}]', **kwargs)

    @staticmethod
    def plot_spectral_number_density(T: Quantity, unit: Unit = u.eV) -> None:
        """Plot the spectral energy density.
        """
        x = Planck.standard_grid(T, unit)
        y = Planck.spectral_number_density(x, T)
        plt.plot(x, y)
        kwargs = dict(xlabel=f'Energy [{unit}]')
        setup_ca(ylabel=f'Spectral number density [{y.unit.to_string("latex")}]', **kwargs)


class CMB(Planck):

    TEMPERATURE_CBE = 2.72548 * u.K

    def __init__(self, temperature=TEMPERATURE_CBE):
        """Overloaded constructor.
        """
        super().__init__(temperature)


class DegenerateFermiGas:

    """Class describing a degenerate Fermi gas.
    """

    @staticmethod
    def fermi_momentum(n: Quantity, g: int = 2) -> Quantity:
        """Return the Fermi momentum at a given number density.

        Arguments
        ---------
        n : Quantity
            The number density of the gas.

        g : int (default 2 for, e.g., eletrons)
            The number of internal degrees of freedom of the costituents.

        Returns
        -------
        Quantity (keV / c)
            The Fermi momentum.
        """
        value = const.h * (3. * n / 4. / np.pi / g)**(1. / 3.)
        return value.to(u.keV_c)

    @staticmethod
    def energy_density_nr(n: Quantity, m: Quantity = const.m_e, g: int = 2) -> Quantity:
        """Return the energy density at a given number density in the non-relativistic
        limit.

        Arguments
        ---------
        n : Quantity
            The number density of the gas.

        m : Quantity (default m_e)
            The mass of the costituent of the gas.

        g : int (default 2 for, e.g., eletrons)
            The number of internal degrees of freedom of the costituents.

        Returns
        -------
        Quantity (eV / cm3)
            The non-relativistic energy density of the gas.
        """
        value = 3. / 10. * (3. / 4. / np.pi / g)**(2. / 3.) * const.h**2. / m * n**(5. / 3.)
        return value.to(u.eV / u.cm**3.)

    @staticmethod
    def pressure_nr(n: Quantity, m: Quantity = const.m_e, g: int = 2) -> Quantity:
        """Return the pressure at a given number density in the non-relativistic
        limit.
        """
        value = 2. / 3. * DegenerateFermiGas.energy_density_nr(n, m, g)
        return value.to(u.bar)

    @staticmethod
    def energy_density_ur(n: Quantity, g: int = 2) -> Quantity:
        """Return the energy density at a given number density in the ultra-relativistic
        limit.

        Arguments
        ---------
        n : Quantity
            The number density of the gas.

        g : int (default 2 for, e.g., eletrons)
            The number of internal degrees of freedom of the costituents.

        Returns
        -------
        Quantity (eV / cm3)
            The non-relativistic energy density of the gas.
        """
        value = 3. / 4. * (3. / 4. / np.pi / g)**(1. / 3.) * const.h * const.c * n**(4. / 3.)
        return value.to(u.eV / u.cm**3.)

    @staticmethod
    def pressure_ur(n: Quantity, g: int = 2) -> Quantity:
        """Return the pressure at a given number density in the ultra-relativistic
        limit.
        """
        value = 1. / 3. * DegenerateFermiGas.energy_density_ur(n, g)
        return value.to(u.bar)
