# Copyright (C) 2025, Luca Baldini (luca.baldini@pi.infn.it)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Test suite for astropart.qstat
"""

from astropy import constants as const
from astropy import units as u
from astropy.units.quantity import Quantity
from astropy.units.core import Unit
import numpy as np
import pytest

from astropart.plot import plt, setup_ca
from astropart.numeric import integral
from astropart.qstat import Planck, CMB, DegenerateFermiGas


def test_integrate_planck_spectral_density(T: Quantity = 5000. * u.K, relative_tolerance: float = 1.e-6):
    """Numerically integrate the Planck spectral energy at a given temperature and
    compare the result with the analytical expression. Note this is done in frequency,
    wavelength and energy space.
    """
    # Cache the target energy density from the analytic expression---note we
    # use pytest.approx() to allow for some numerical inaccuracy in the integrals.
    u_target = pytest.approx(Planck.energy_density(T).value, relative_tolerance)
    # Since we are at it, test the expression for the pressure.
    assert Planck.pressure(T).to(u.eV / u.cm**3).value * 3. == u_target
    # Calculation in frequency space.
    du = lambda x: Planck.spectral_energy_density(x * u.THz, T).value
    assert integral(du, 1., 10000.) == u_target
    # Calculation in wavelength space.
    du = lambda x: Planck.spectral_energy_density(x * u.nm, T).value
    assert integral(du, 10., 1000000.) == u_target
    # Calculation in energy space.
    du = lambda x: Planck.spectral_energy_density(x * u.eV, T).value
    assert integral(du, 0.01, 100.) == u_target


def test_frozen_planck(T: Quantity = 5000. * u.K):
    """Test that a frozen Planck distribution behaves exaclty as the general
    one at the correct temperature.
    """
    planck = Planck(T)
    nu = Planck.standard_grid(T, u.nm)
    assert (planck.spectral_energy_density(nu) == Planck.spectral_energy_density(nu, T)).all()
    assert planck.energy_density() == Planck.energy_density(T)
    assert planck.number_density() == Planck.number_density(T)


def test_cmb():
    """Test the basic figures for the CMB.
    """
    cmb = CMB()
    plt.figure('CMB spectral density')
    cmb.plot_spectral_energy_density(u.eV)
    E = cmb.peak_energy()
    plt.plot(E, cmb.spectral_energy_density(E), 'o')
    plt.axvline(cmb.average_energy().value, ls='dashed')
    assert cmb.energy_density() == 0.26056292996720726 * u.eV / u.cm**3
    assert cmb.peak_energy() == 0.0006626538224727968 * u.eV
    assert cmb.average_energy() == 0.0006344138118349207 * u.eV
    assert cmb.number_density() == 410.71780612332697 / u.cm**3


def test_plot_planck(T: Quantity = 5000. * u.K):
    """Basic test for plotting the Planck distribution.
    """
    plt.figure('Planck lambda')
    Planck.plot_spectral_energy_density(T, u.nm)
    plt.figure('Planck nu')
    Planck.plot_spectral_energy_density(T, u.THz)
    plt.figure('Planck E')
    Planck.plot_spectral_energy_density(T, u.eV)


def test_fermi_gas():
    """Tests for the degenerate Fermi gas.
    """
    n = np.geomspace(1.e26, 1.e33, 100) * u.cm**-3

    plt.figure('Fermi momentum')
    plt.plot(n, DegenerateFermiGas.fermi_momentum(n))
    plt.axhline((const.m_e * const.c**2).to(u.keV).value, ls='dashed')
    setup_ca(xlabel='$n$ [cm$^{-3}$]', ylabel='$p_F$ [keV / c]', logx=True, logy=True)

    plt.figure('Electron degeneracy energy density')
    plt.plot(n, DegenerateFermiGas.energy_density_nr(n), label='non-relativistic')
    plt.plot(n, DegenerateFermiGas.energy_density_ur(n), label='ultra-relativistic')
    setup_ca(xlabel='n [cm$^{-3}$]', ylabel='u [eV cm$^{-3}$]', logx=True, logy=True, legend=True)

    plt.figure('Electron degeneracy pressure')
    plt.plot(n, DegenerateFermiGas.pressure_nr(n), label='non-relativistic')
    plt.plot(n, DegenerateFermiGas.pressure_ur(n), label='ultra-relativistic')
    setup_ca(xlabel='n [cm$^{-3}$]', ylabel='P [bar]', logx=True, logy=True, legend=True)



if __name__ == '__main__':
    test_plot_planck()
    test_cmb()
    test_fermi_gas()
    plt.show()
