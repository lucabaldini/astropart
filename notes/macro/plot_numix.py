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

"""Plot some relevant quantities for the solar neutrinos.
"""

import astropy.constants as const
import astropy.units as u
import numpy as np
from scipy.stats import norm

from astropart import numix
from astropart.ssm import BS05
from astropart.plot import plt, margin_figure, standard_figure, setup_ca, save_cf


margin_figure('nu_oscillation_length', height=2.5)
E = np.geomspace(0.2, 5e3)
L21 = numix.oscillation_length_21(E)
L32 = numix.oscillation_length_32(E)
plt.plot(E, L21, color='black', ls='dashed')
x = 1.e2
y = numix.oscillation_length_21(x)
plt.text(x, y, '$\\Delta m^2_\\odot$', ha='right', va='bottom', size='small')
plt.plot(E, L32, color='black', ls='dashed')
x = 1.e1
y = numix.oscillation_length_32(x)
plt.text(x, y, '$\\Delta m^2_\\mathrm{atm}$', ha='left', va='top', size='small')
setup_ca(xlabel='Energy [MeV]', ylabel='$L_\\mathrm{osc}$ [m]', logx=True, logy=True,
    ymin=10., ymax=1.e7, xticks=np.logspace(0., 3., 4), yticks=np.logspace(1., 7., 7))
save_cf()


standard_figure('sun neutrino oscillation', height=2.5)
# Put all of this into numix.py
L = np.linspace(0., 200, 250)
E = 1.
eres = 0.1
phase = 1.27 * numix.DELTA_M21_CBE * 1000. * L / E
prob = np.sin(2. * np.radians(numix.THETA_12_CBE))**2. * np.sin(phase)**2.
plt.plot(L, prob, color='gray', ls='dashed')
plt.text(80., 0.875, 'Monochromatic 1 MeV beam', color='gray')
mu = E
sigma = E * eres
num_sigma = 5.
num_points = 1000
x = np.linspace(mu - num_sigma * sigma, mu + num_sigma * sigma, num_points)
dx = 2. * num_sigma / num_points
pdf = norm(mu, sigma).pdf(x)
prob = []
for l in L:
    phase = 1.27 * numix.DELTA_M21_CBE * 1000. * l / x
    P = np.sin(2. * np.radians(numix.THETA_12_CBE))**2. * np.sin(phase)**2.
    ave = np.sum(P * pdf * dx) / np.sum(pdf * dx)
    prob.append(ave)
plt.plot(L, prob, color='black')
plt.axhline(0.5 * np.sin(2. * np.radians(numix.THETA_12_CBE))**2., color='black', ls='dashed')
plt.text(150, 0.45, '$\\sigma_E / E = 10\\%$', ha='center')
setup_ca(xlabel='$L$ [km]', ylabel='Oscillation probability', ymax=1., grids=False)
save_cf()


margin_figure('msw conversion', height=2.5, left=0.2)
zeta = np.linspace(0., 2.5, 100)
plt.hlines(numix.DELTA_M21_CBE, 0., 1., color='black')
plt.hlines(numix.DELTA_M21_CBE, 1., 2.5, color='black', ls='dashed')
mask = zeta < 1
plt.plot(zeta[mask], zeta[mask] * numix.DELTA_M21_CBE, color='black', ls='dashed')
mask = zeta >= 1
plt.plot(zeta[mask], zeta[mask] * numix.DELTA_M21_CBE, color='black')
plt.text(2., numix.DELTA_M21_CBE, '$\\nu_\\mu$', va='bottom')
plt.text(2., 2.3 * numix.DELTA_M21_CBE, '$\\nu_e$', va='bottom')
plt.text(0.1, 2.4 * numix.DELTA_M21_CBE, '$\\rightarrow$ increasing $\\varrho$')
plt.plot(1., numix.DELTA_M21_CBE, 'o', color='black', ms=4)
plt.text(1., 0.95 * numix.DELTA_M21_CBE, 'Resonant conversion', ha='left', va='top', size='x-small')
setup_ca(xlabel='$\\zeta$', ylabel='$m^2$', yticks=(), xticks=(0., 1., 2.))
save_cf()

margin_figure('msw resonance energy', height=2.5)
r = np.linspace(0., 0.275, 100)
ne = BS05.electron_numer_density(r)
E = numix.msw_resonance_energy(ne) * 1.e-6
plt.plot(r, E, color='black')
plt.plot(r, 1000. * BS05.pdf_flux_B8(r), color='gray', ls='dashed')
setup_ca(xlabel='$r$ [$R_\\odot$]', ylabel='$E_c$ [MeV]', xticks=(0., 0.1, 0.2),
    ymin=0.)
save_cf()


print(numix.DELTA_M21_CBE / 2. / np.sqrt(2.) / numix.G_F / 1.e6)
print(numix.msw_resonance_energy(BS05.electron_numer_density(0.)))

plt.show()
