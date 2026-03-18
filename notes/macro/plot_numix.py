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


margin_figure('msw conversion', height=2.5, left=0.225)
delta_m2 = numix.DELTA_M21_CBE
m1sq = 0
m2sq = delta_m2
zeta = np.linspace(0., 4., 100)
plt.axhline(numix.DELTA_M21_CBE, color='black')
plt.plot(zeta, zeta * numix.DELTA_M21_CBE, color='black')
plt.text(2., 1.1 * numix.DELTA_M21_CBE, '$\\nu_\\mu = \\nu_2$', va='bottom')
plt.text(2.6, 2.3 * numix.DELTA_M21_CBE, '$\\nu_e = \\nu_1$', va='bottom')
plt.text(0.1, 3.5 * numix.DELTA_M21_CBE, '$\\rightarrow$ increasing $\\varrho$')
plt.plot(1., numix.DELTA_M21_CBE, 'o', color='black', ms=4)
plt.text(1., 0.9 * numix.DELTA_M21_CBE, 'Resonant conversion',
         ha='left', va='top', size='x-small')
setup_ca(xlabel='$\\zeta$', ylabel='$\\tilde{m}^2_i$', yticks=(m1sq, m2sq), ymin=0.,
         xticks=(0., 1., 2., 3., 4.))
plt.gca().yaxis.set_label_coords(-0.12, 0.5)
plt.gca().set_yticklabels(('$m_1^2$', '$m_2^2$'))
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


margin_figure('msw mixing', height=2.5, left=0.275)
zeta = np.linspace(0., 4, 100)
c2t = np.cos(2. * np.radians(numix.THETA_12_CBE))
s2t = np.sin(2. * np.radians(numix.THETA_12_CBE))
C = np.sqrt((c2t - zeta)**2. + (s2t)**2.)
theta = 0.5 * np.atan2(s2t, c2t - zeta)
plt.plot(zeta, np.degrees(theta), color='black')
plt.axhline(numix.THETA_12_CBE, color='black', ls='dashed')
plt.text(1.5, numix.THETA_12_CBE + 1., '$\\theta$ in vacuum', ha='left', va='bottom')
setup_ca(xlabel='$\\zeta$', ylabel='$\\tilde{\\theta}$', ymin=0., ymax=90.,
         xticks=(0., 1., 2., 3., 4.))
save_cf()


margin_figure('msw mass', height=2.5, left=0.225)
zeta = np.linspace(0., 4, 100)
C = np.sqrt((c2t - zeta)**2. + (s2t)**2.)
delta_m2 = numix.DELTA_M21_CBE
m1sq = 0
m2sq = delta_m2
m1_tilde2 = 0.5 * (m1sq + m2sq + zeta * delta_m2 - C * delta_m2)
m2_tilde2 = 0.5 * (m1sq + m2sq + zeta * delta_m2 + C * delta_m2)
plt.plot(zeta, m1_tilde2, color='black')
plt.plot(zeta, m2_tilde2, color='black')
plt.text(0.1, 3.75 * numix.DELTA_M21_CBE, '$\\rightarrow$ increasing $\\varrho$')
plt.text(2., m2sq - 2.e-5, "$\\nu_1$")
plt.text(2.5, m2sq + 1.2e-4, "$\\nu_2$")
setup_ca(xlabel='$\\zeta$', ylabel='$\\tilde{m}^2_i$', yticks=(m1sq, m2sq),
         ymin=0., xticks=(0., 1., 2., 3., 4.))
plt.gca().yaxis.set_label_coords(-0.12, 0.5)
plt.gca().set_yticklabels(('$m_1^2$', '$m_2^2$'))
save_cf()


print(numix.DELTA_M21_CBE / 2. / np.sqrt(2.) / numix.G_F / 1.e6)
print(numix.msw_resonance_energy(BS05.electron_numer_density(0.)))

plt.show()
