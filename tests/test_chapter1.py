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

"""Proofreading for Chapter 1.
"""

from loguru import logger

import astropy.constants as const
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import numpy as np

from astropart.ns import CanonicalPulsar
from astropart.qstat import DegenerateFermiGas, Planck
from astropart.wd import UniformWhiteDwarf, WhiteDwarf


def test_chapter():
    """Proofread chapter 1.
    """
    logger.info('Introduction...')

    # Calculation of the pc.
    pc_au_ratio = 3600. * 180. / np.pi
    pc = pc_au_ratio * u.au
    assert pc == 1. * u.pc

    # Print basic units.
    print(f'1 au = {1. * u.au.to(u.m):.4e} m')
    print(f'1 lyr = {1. * u.lyr.to(u.m):.4e} m')
    print(f'1 pc = {1. * u.pc.to(u.m):.4e} m')

    # Get a rough idea of the arcsec---it is 5 mm at a distance of 1 km.
    print(f'1 au = {1. / pc_au_ratio} pc')
    print(f'1 pc = {pc_au_ratio:.4e} au')


    logger.info('Hubble law...')

    H0 = cosmo.H(0)
    hubble_time = 1. / H0
    hubble_radius = const.c / H0
    critical_density = 3 * H0**2 / 8. / np.pi / const.G
    print(f'H0 = {H0}')
    print(f'Hubble time = {hubble_time.to(u.gigayear)}')
    print(f'Hubble radius = {hubble_radius.to(u.Gpc)}')
    print(f'Critical density = {critical_density.to(u.g / u.cm**3)}')
    print(f'Critical density = {(critical_density * const.c**2).to(u.keV / u.cm**3)}')


    logger.info('Earth atmosphere...')
    T0 = 240. * u.K
    mu = 29. * u.g / u.mol
    g = 9.81 * u.m / u.s**2.
    P0 = 1.013 * u.bar
    rho0 = 1.24 * u.mg / u.cm**3.
    scale_height = (const.R * T0 / g / mu).to(u.km)
    grammage = (rho0 * scale_height).to(u.g / u.cm**2.)
    rho_lead = 11.35 * u.g / u.cm**3.
    print(f'Isothermal model at T = {T0} and mu = {mu}')
    print(f'Pressure at sea level: {P0}')
    print(f'Density at sea level: {rho0}')
    print(f'Atmospheric scale height = {scale_height}')
    print(f'Grammage at sea level = {grammage}')
    print(f'Grammage lead equivalent = {grammage / rho_lead}')


    logger.info('Earth magnetic field...')


    logger.info('Sun inner model...')
    atm = u.def_unit("atm", 101325 * u.Pa)
    P_core = (32. / np.pi * const.G * const.M_sun**2. / const.R_sun**4).to(atm)
    rho_core = (24. / np.pi * const.M_sun / const.R_sun**3).to(u.g / u.cm**3.)
    T_core = (2. / 3. * const.m_p * const.G * const.M_sun / const.k_B / const.R_sun).to(u.K)
    P_rad = (4 * const.sigma_sb / 3. / const.c * T_core**4).to(atm)
    print(f'Sun core pressure = {P_core:.4e}')
    print(f'Sun core density = {rho_core}')
    print(f'Sun core temperature = {T_core:.4e}')
    print(f'Sun radiation pressure at the core = {P_rad:.4e}')
    print(f'P_rad / P_core = {P_rad / P_core:.4e}')


    logger.info('White dwarfs...')
    M = const.M_sun
    n = UniformWhiteDwarf.non_relativistic_stoner_number_density(M)
    rho = n * UniformWhiteDwarf.REFERENCE_MU * const.m_p
    pF = DegenerateFermiGas.fermi_momentum(n)
    R = UniformWhiteDwarf.non_relativistic_stoner_radius(M)
    print(f'Reference mass = {M / const.M_sun} solar masses')
    print(f'Stoner number density = {n}')
    print(f'Stoner mass density = {rho}')
    print(f'Stoner radius = {R}')
    print(f'Stoner Fermi momentum = {pF}')
    print(f'Stoner limit: {UniformWhiteDwarf.stoner_limit() / const.M_sun} solar masses')
    print(f'Chandrasekar limit: {WhiteDwarf.chandrasekhar_limit() / const.M_sun} solar masses')


    logger.info('Supernovae...')
    Q = 1. * u.MeV
    E_Ia = (const.M_sun / const.m_p * Q).to(u.erg)
    v_ej = np.sqrt(2. * E_Ia / const.M_sun).to(u.km / u.s)
    print(f'SN Ia energy release = {E_Ia}')
    print(f'SN Ia ejecta velocity = {v_ej}')


    logger.info('Neutron stars...')
    R_ratio = const.m_e / const.m_n * UniformWhiteDwarf.REFERENCE_MU**(5. / 3)
    R_WD = const.R_earth
    R_NS = (R_WD * R_ratio).to(u.km)
    A_NS = 4. * np.pi * R_NS**2
    M = const.M_sun
    rho_NS = (3. * M / 4. / np.pi / R_NS**3.).to(u.g / u.cm**3)
    g_NS = (const.G * M / R_NS**2).to(u.m / u.s**2)
    v_escape = np.sqrt(2. * const.G * M / R_NS).to(u.km / u.s) / const.c.to(u.km / u.s)
    P = 1. * u.ms
    v_radial = (2. * np.pi * R_NS / P).to(u.km / u.s) / const.c.to(u.km / u.s)
    T = 1.e6 * u.K
    L = (A_NS * const.sigma_sb * T**4).to(u.W) / const.L_sun.to(u.W)
    print(f'NS / WD radius ratio = {R_ratio}')
    print(f'NS radius = {R_NS}')
    print(f'NS density = {rho_NS:.3e}')
    print(f'NS surface gravity = {g_NS:.3e}')
    print(f'NS Newtonian escape velocity = {v_escape:.2f} c')
    print(f'NS equatorial velocity at P = {P} = {v_radial:.2f} c')
    print(f'NS luminosity at {T} = {L:.3f} L_sun')
    print(f'Peak energy at {T}: {Planck.peak_energy(T)}')


    logger.info('Pulsars...')
    P = 1.3 * u.s
    print(f'Minimum pulsar density @ {P} = {CanonicalPulsar.minimum_density(P):.3e}')
    P = 33.1 * u.ms
    print(f'Minimum pulsar density @ {P} = {CanonicalPulsar.minimum_density(P):.3e}')
    P = 1.4 * u.ms
    print(f'Minimum pulsar density @ {P} = {CanonicalPulsar.minimum_density(P):.3e}')
    P_crab = 0.033392 * u.s
    Pdot_crab = 4.21e-13
    print(f'Crab P = {P_crab}, pdot = {Pdot_crab}')
    E_crab = CanonicalPulsar.rotational_energy(P_crab)
    B_crab = CanonicalPulsar.characteristic_magnetic_field(P_crab, Pdot_crab)
    age_crab = CanonicalPulsar.characteristic_age(P_crab, Pdot_crab)
    print(f'Crab rotational energy = {E_crab:.3e}')
    print(f'Crab characteristic B = {B_crab:.3e}')
    print(f'Crab characteristic age = {age_crab:.3e}')