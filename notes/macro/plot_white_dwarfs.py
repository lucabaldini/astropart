# Copyright (C) 2023, Luca Baldini (luca.baldini@pi.infn.it)
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

"""Plot some relevant quantities specific to white dwarfs.
"""

import math

import astropy.constants as const
import astropy.units as u
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from plotting import plt, standard_figure, margin_figure, setup_ca, save_cf


h = const.h.cgs
c = const.c.cgs
g_e = 2
G = const.G.cgs
m_H = 1.00784 * 0.5 * (const.m_p.cgs + const.m_p.cgs)
m_e = const.m_e.cgs
M_s = const.M_sun.cgs
R_s = const.R_sun.cgs

REFERENCE_MU = 2.
STONER_MU = 2.5
M_SIRIUS_B = 0.85 * M_s
M_ERIDANI_B = 0.44 * M_s

# Table III in https://articles.adsabs.harvard.edu/pdf/1935MNRAS..95..207C
# These are calculated for mu = 1. For a generic mu, masses should be divided
# by mu**2, and radii by mu.
CHANDRA_NORM_M = np.array([
    0.877, 1.612, 2.007, 2.440, 2.934, 3.528, 4.310, 4.852, 5.294, 5.484, 5.728]
    )
CHANDRA_NORM_R = 1.e9 * np.array([
    2.792, 2.155, 1.929, 1.721, 1.514, 1.287, 0.9936, 0.7699, 0.5443, 0.4136, 0.
])


def p_F(n):
    """Return the Fermi momentum for a given electron number density.
    """
    return (3. * h**3. * n / g_e / 4. / np.pi)**(1. / 3.)

def uniform_sphere_radius(M, n, mu=REFERENCE_MU):
    """Return the stellar radius as a function of total mass and number density
    for a uniform sphere.
    """
    return (3. * M / 4. / np.pi / n / mu / m_H)**(1. / 3.)

def stoner_n_nr(M, mu=REFERENCE_MU):
    """Return the non-relativistic equilibrium volumetric density for a given
    stellar mass following Stoner (1929).
    """
    return 256. * (np.pi / 3.)**3. * G**3. * (mu * m_H)**4. * m_e**3. / h**6 * M**2.

def stoner_radius_nr(M, mu=REFERENCE_MU):
    """Return the stellar radius in the non-relativistic limit following Stoner (1929).
    """
    return (3. / np.pi)**(4. / 3.) / 2.**(10. / 3.) *\
        h**2. / G / (mu * m_H)**(5. / 3.) / m_e / M**(1. / 3.)

def stoner_rho_nr(M, mu=REFERENCE_MU):
    """Return the non-relativistic equilibrium mass density for a given stellar
    mass following Stoner (1929).
    """
    return mu * m_H * stoner_n_nr(M, mu)

def F(x):
    """Equation (19) in Stoner (1930).

    x = p_F / (m_e c) is the (adimensional) normalized Fermi momentum.
    """
    return 1. / (8. * x**3.) * (
        3. / x * np.log(x + np.sqrt(1. + x**2.)) +
        np.sqrt(1. + x**2) * (2. * x**2 - 3)
    )

def stoner_mass(n, mu=REFERENCE_MU):
    """Re-implementation of equation (18a) in Stoner (1930), in terms of all the
    relevant quantities (i.e., with no pre-calculated numerical factors).
    """
    x = (p_F(n) / m_e / c).value
    C = (5. * h * c / g_e**(1. / 3.) / G / (mu * m_H)**(4. / 3.)) * (3. / 4. / np.pi)**(2. / 3.)
    return (C * F(x))**(3. / 2.)

def stoner_mass_limit(mu=REFERENCE_MU):
    """Return the maximim white dwarf mass in the unfiform-sphere approximation
    following Stoner (1930).
    """
    return (3. / 4. / np.pi) * (5. / 4.)**(3. / 2.) *\
        (h * c / G)**(3. / 2.) / g_e**(1. / 2.) / (mu * m_H)**2.

def chandrasekar_mass_limit(mu=REFERENCE_MU):
    """Return the famous Chadrasekar limit.
    """
    return 2.018236 * np.sqrt(3. * np.pi) / 2. * (h * c / 2. / np.pi / G)**(3. / 2.) /\
        (mu * m_H)**2


# A few consistency checks with Stoner (1929). Unfortunately, as pointed out in
# Stoner (1930), there is a slight mistake in one of the constants, and most of
# the figures are slightly off.
print(f'--- All the following figures assume mu = {STONER_MU}!')
print(f'n reference: {stoner_n_nr(M_s, STONER_MU) / M_s**2.:.3e} (2.31e-37)')
print(f'Max density for Sirius B: {stoner_rho_nr(M_SIRIUS_B, STONER_MU):.3e} (2.77e6)')
print(f'Core radius for Sirius B: {stoner_radius_nr(M_SIRIUS_B, STONER_MU).to(R_s):.3e} (0.0075)')
print(f'Max density for Eridani B: {stoner_rho_nr(M_ERIDANI_B, STONER_MU):.3e} (7.48e5)')
print(f'Core radius for Eridani B: {stoner_radius_nr(M_ERIDANI_B, STONER_MU).to(R_s):.3e} (0.011)')
print(f'Stoner mass limit: {stoner_mass_limit(STONER_MU).to(M_s)} ({2.19e33 / M_s})')

# And more consitency checks with Stoner (1930)
# This is from table I, x = 1.
n0 = 10**29.7690
M0 = 10**33.0233
print(f'Stoner mass for x = 1: {stoner_mass(n0, STONER_MU)} ({M0})')

# And, finally, Chandrasekar.
print(f'Chandrasekar mass limit: {chandrasekar_mass_limit(STONER_MU).to(M_s)} (0.91)')
print(f'--- End of figures assume mu = {STONER_MU}!')

# A few figures for the main text.
print(f'Non-relativistic number density for 1 solar mass: {stoner_n_nr(M_s):.3e}')
print(f'Non-relativistic mass density for 1 solar mass: {stoner_rho_nr(M_s):.3e}')
print(f'Non-relativistic fermi momentum for 1 solar mass: {p_F(stoner_n_nr(M_s)).to(u.MeV / c)}')
print(f'Stoner mass limit: {stoner_mass_limit().to(M_s)}')
print(f'Chandrasekar mass limit: {chandrasekar_mass_limit().to(M_s)}')


M = np.geomspace(0.05 * M_s, 2. * M_s)
n = stoner_n_nr(M)
rho = stoner_rho_nr(M)
R = uniform_sphere_radius(M, n)
p = p_F(n)

standard_figure('white dwarf density nr')
plt.plot(M.to(M_s), rho, color='k')
setup_ca(xlabel='Stellar mass $[M_\\odot]$', ylabel='Density $[g cm^{-3}]$', logx=True, logy=True)

standard_figure('white dwarf radius nr')
plt.plot(M.to(M_s), R.to(R_s), color='k')
setup_ca(xlabel='Stellar mass $[M_\\odot]$', ylabel='Density $[g cm^{-3}]$', logx=True, logy=True)

margin_figure('white dwarf fermip nr', height=2.25, left=0.25, right=0.85, bottom=0.25)
plt.plot(M.to(M_s), p.to(u.MeV / c), color='k')
plt.axhline(0.511, color='k', ls='dashed')
plt.text(0.275, 0.511, 'Electron mass', size='small', va='bottom')
setup_ca(xlabel='Stellar mass $[M_\\odot]$', ylabel='Non-relativistic $p_F$ [MeV/c]')
save_cf()

standard_figure('stoner 1930 figure 1')
logn = np.linspace(28., 32., 100)
m = stoner_mass(10**logn, STONER_MU)
plt.plot(logn, np.log10(m.value), color='k')
setup_ca(xlabel='log n', ylabel='log M')

standard_figure('stoner function')
x = np.geomspace(0.1, 100., 100)
y = F(x)
plt.plot(x, F(x), color='k')
setup_ca(logx=True, xlabel='x', ylabel='F(x)')

standard_figure('white dwarf mass radius')
n = np.logspace(25., 45., 100)
M = stoner_mass(n)
R = uniform_sphere_radius(M, n)
plt.plot(M.to(M_s), R / R_s, color='k', label='Stoner (1930)')
M = np.geomspace(0.05 * M_s, 2. * M_s)
R_nr = stoner_radius_nr(M)
plt.plot(M.to(M_s), R_nr.to(R_s), color='gray', label='Non-relativistic limit')
y = const.R_earth.cgs / R_s
plt.axhline(y, color='k', ls='dashed')
plt.text(0.15, y, 'Earth radius', size='small', va='bottom')
#plt.text(1.5, 0.005, 'Non relativistic', color='gray', ha='center', size='small')
#plt.text(1.1, 0.0015, 'Relativistic', color='k', ha='left', size='small')
M = CHANDRA_NORM_M / REFERENCE_MU**2.
R = CHANDRA_NORM_R / REFERENCE_MU / R_s
s = InterpolatedUnivariateSpline(M, R, k=3)
x = np.linspace(M.min(), M.max(), 100)
plt.plot(x, s(x), color='k')
plt.plot(M, R, 'o', color='k', fillstyle='none', ms=3.5, label='Chandrasekar (1935)')
setup_ca(xmin=0.1, xmax=2., ymin=0., ymax=0.02, xlabel='Stellar mass [$M_\\odot$]',
    ylabel='Stellar radius [$R_\\odot$]', legend=True)
save_cf()


if __name__ == '__main__':
    plt.show()
