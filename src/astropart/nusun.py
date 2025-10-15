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

"""Neutrino stuff.
"""

import astropy.constants as const
import astropy.units as u
import numpy as np

from astropart import ASTROPART_DATA
from astropart.numeric import Spline
from astropart import amdc


# Data are from https://www.annualreviews.org/doi/pdf/10.1146/annurev-nucl-011921-061243
# Table 1, column Solar (global), with LC whenever it makes sense, falling back
# to the calculations based on the SSM when only upper limits are available.
FLUX_PP = 5.971e10
FLUX_PEP = 1.448e8
ENERGY_PEP = 1.442
FLUX_HEP = 7.98e3
FLUX_BE7 = 4.80e9
ENERGIES_BE7 = (0.3843, 0.8613)
YIELDS_BE7 = (0.102, 0.898)
FLUX_B8 = 5.16e6
FLUX_N13 = 2.78e8
FLUX_O15 = 2.5e8
FLUX_F17 = 5.29e6
FLUX_ECN = 2.370e5
ENERGY_ECN = 2.218
FLUX_ECO = 8.205e4
ENERGY_ECO = 2.751

# Calculate some intermediate quantities.
FLUX_PP_CYCLE = FLUX_PP + FLUX_PEP + FLUX_HEP + FLUX_BE7 + FLUX_B8
FLUX_CNO_CYCLE = FLUX_N13 + FLUX_O15 + FLUX_F17 + FLUX_ECN + FLUX_ECO
FLUX_TOTAL = FLUX_PP_CYCLE + FLUX_CNO_CYCLE

# This controls the units on the y-axis on the fluxes, with 0.1 being appropriate
# for cm^{-2} s^{-1} 100~keV^{-1} and 1. for cm^{-2} s^{-1} MeV^{-1}.
FLUX_SCALE = .1

# Load all the spectra from the pdfs available at http://www.sns.ias.edu/~jnb/
spec_pp = Spline.from_file(ASTROPART_DATA / 'nusun_spec_pp.txt', yscale=FLUX_SCALE * FLUX_PP)
spec_hep = Spline.from_file(ASTROPART_DATA / 'nusun_spec_hep.txt', yscale=FLUX_SCALE * FLUX_HEP)
spec_b8 = Spline.from_file(ASTROPART_DATA / 'nusun_spec_B8.txt', yscale=FLUX_SCALE * FLUX_B8)
spec_n13 = Spline.from_file(ASTROPART_DATA / 'nusun_spec_N13.txt', yscale=FLUX_SCALE * FLUX_N13)
spec_o15 = Spline.from_file(ASTROPART_DATA / 'nusun_spec_O15.txt', yscale=FLUX_SCALE * FLUX_O15)
spec_f17 = Spline.from_file(ASTROPART_DATA / 'nusun_spec_F17.txt', yscale=FLUX_SCALE * FLUX_F17)


def coulomb_barrier(Z1: int, A1: int, Z2: int, A2: int) -> float:
    """Return the height of the Coulomb barrier (in MeV) for a nucleus-nucleus
    thermonuclear fusion process.

    Note we have a slight ambiguity, here, as to what we should use for r_0, which
    we take to be max(A1, A2)**(1. / 3.) * u.fm.

    Parameters
    ----------
    Z1 : int
        The atomic number of the first nucleus.

    A1 : int
        The mass number of the first nucleus.

    Z2 : int
        The atomic number of the second nucleus.

    A2 : int
        The mass number of the second nucleus.
    """
    r0 = max(A1, A2)**(1. / 3.) * u.fm
    return (Z1 * Z2 * const.e.gauss**2. / r0).to_value(u.MeV)


def gamow_energy(Z1: int, A1: int, Z2: int, A2: int) -> float:
    """Return the Gamow energy (in MeV) for a nucleus-nucleus thermonuclear fusion process.

    Parameters
    ----------
    Z1 : int
        The atomic number of the first nucleus.

    A1 : int
        The mass number of the first nucleus.

    Z2 : int
        The atomic number of the second nucleus.

    A2 : int
        The mass number of the second nucleus.
    """
    data1 = amdc.nuclear_mass_data(Z1, A1)
    data2 = amdc.nuclear_mass_data(Z2, A2)
    m1 = data1.atomic_mass
    m2 = data2.atomic_mass
    mu = m1 * m2 / (m1 + m2) * u.u
    return (2. * mu * np.pi**2. * const.e.gauss**4 * Z1**2. * Z2**2. / const.hbar**2).to_value(u.MeV)


def tunnel_probability(Z1: int, A1: int, Z2: int, A2: int, E: float) -> float:
    """Return the quantum tunneling probability in WKB approximationfor a nucleus-nucleus
    thermonuclear fusion process.

    Parameters
    ----------
    Z1 : int
        The atomic number of the first nucleus.

    A1 : int
        The mass number of the first nucleus.

    Z2 : int
        The atomic number of the second nucleus.

    A2 : int
        The mass number of the second nucleus.

    E : float
        The energy (in MeV) of the incoming nucleus.
    """
    E_G = gamow_energy(Z1, A1, Z2, A2)
    return np.exp(-np.sqrt(E_G / E))


def gamow_peak_mode(Z1: int, A1: int, Z2: int, A2: int, kT: float) -> float:
    """Return the energy (in MeV) where the Gamow peak has its maximum a nucleus-nucleus
    thermonuclear fusion process.

    Parameters
    ----------
    Z1 : int
        The atomic number of the first nucleus.

    A1 : int
        The mass number of the first nucleus.

    Z2 : int
        The atomic number of the second nucleus.

    A2 : int
        The mass number of the second nucleus.

    kT : float
        The equivalent thermal kinetic energy in MeV.
    """
    E_G = gamow_energy(Z1, A1, Z2, A2)
    return E_G**(1. / 3.) * (kT / 2.)**(2. / 3)


def gamow_peak_sigma(Z1: int, A1: int, Z2: int, A2: int, kT: float) -> float:
    """Return the equivalent sigma (in MeV) of the Gamow peak in Gaussian approximation
    for a nucleus-nucleus thermonuclear fusion process.

    Parameters
    ----------
    Z1 : int
        The atomic number of the first nucleus.

    A1 : int
        The mass number of the first nucleus.

    Z2 : int
        The atomic number of the second nucleus.

    A2 : int
        The mass number of the second nucleus.

    kT : float
        The equivalent thermal kinetic energy in MeV.
    """
    E0 = gamow_peak_mode(Z1, A1, Z2, A2, kT)
    return 2. / np.sqrt(3.) * (E0 * kT)**0.5


def tau(Z1: int, A1: int, Z2: int, A2: int, kT: float) -> float:
    """Return the tau parameter for a nucleus-nucleus thermonuclear fusion process.

    Parameters
    ----------
    Z1 : int
        The atomic number of the first nucleus.

    A1 : int
        The mass number of the first nucleus.

    Z2 : int
        The atomic number of the second nucleus.

    A2 : int
        The mass number of the second nucleus.

    kT : float
        The equivalent thermal kinetic energy in MeV.
    """
    E0 = gamow_peak_mode(Z1, A1, Z2, A2, kT)
    return 3. * E0 / kT


def temperature_index(Z1: int, A1: int, Z2: int, A2: int, kT: float) -> float:
    """Return the apprximate power-law index for the temperature scaling for
    a nucleus-nucleus thermonuclear fusion process.

    Parameters
    ----------
    Z1 : int
        The atomic number of the first nucleus.

    A1 : int
        The mass number of the first nucleus.

    Z2 : int
        The atomic number of the second nucleus.

    A2 : int
        The mass number of the second nucleus.

    kT : float
        The equivalent thermal kinetic energy in MeV.
    """
    return (tau(Z1, A1, Z2, A2, kT) - 2.) / 3.



if __name__ == '__main__':
    T = 14.e6 * u.K
    print((const.k_B * T).to(u.keV))
    kT = 0.0012064
    PP_CYCLE_NUCLEAR_REACTIONS = ((1, 1, 1, 1), (1, 2, 1, 1), (2, 3, 2, 3), (2, 4, 2, 3), (4, 7, 1, 1))
    for Z1, A1, Z2, A2 in PP_CYCLE_NUCLEAR_REACTIONS:
        print(f'reaction ({Z1}, {A1}) + ({Z2}, {A2})...')
        print(f'Coulomb barrier: {coulomb_barrier(Z1, A1, Z2, A2):.3f} MeV')
        print(f'Gamow energy: {gamow_energy(Z1, A1, Z2, A2):.3f} MeV')
        print(f'Tunnel probability at kT = {1000. * kT:.2f} keV: {tunnel_probability(Z1, A1, Z2, A2, kT):.3e}')
        print(f'E0 at kT = {1000. * kT:.2f} keV: {1000. * gamow_peak_mode(Z1, A1, Z2, A2, kT):.3f} keV')
        print(f'sigma0 at kT = {1000. * kT:.2f} keV: {1000. * gamow_peak_sigma(Z1, A1, Z2, A2, kT):.3f} keV')
        print(f'tau at kT = {1000. * kT:.2f} keV: {tau(Z1, A1, Z2, A2, kT):.3f}')
        print(f'temperature index at kT = {1000. * kT:.2f} keV: {temperature_index(Z1, A1, Z2, A2, kT):.3f}')
        print()
