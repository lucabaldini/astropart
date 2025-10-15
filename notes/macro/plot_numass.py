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

"""Plot some relevant quantities for meutrino masses.
"""

import astropy.constants as const
import astropy.units as u
import numpy as np

from astropart.numix import DELTA_M21_CBE, DELTA_M32_CBE
from astropart.plot import plt, margin_figure, setup_ca, save_cf


margin_figure('neutrino mass scale', height=2.25)
m1 = np.geomspace(0.001, 0.3, 100)
m2 = np.sqrt(DELTA_M21_CBE + m1**2.)
m3_no = np.sqrt(DELTA_M32_CBE + m2**2.)
m3_io = np.sqrt(-DELTA_M32_CBE + m2**2.)
plt.plot(m1, m1, color='black')
plt.plot(m1, m2, color='black')
plt.plot(m1, m3_no, color='black')
plt.plot(m1, m3_io, color='black', ls='dashed')
plt.text(0.002, 0.0055, '$m_1$')
plt.text(0.002, 0.012, '$m_2$')
plt.text(0.002, 0.06, '$m_3$ (NO)')
plt.text(0.03, 0.007, '$m_3$ (IO)')
setup_ca(xlabel='$m_1$ [eV]', ylabel='$m_i$ [eV]', logx=True, logy=True, ymin=5.e-3, ymax=m1.max())
save_cf()

print('Solar neutrinos...')
E = 1. * u.MeV
L = 1. * u.au
for delta in (DELTA_M21_CBE, DELTA_M32_CBE):
    phi = delta * (L / (4 * const.hbar * const.c * E)).to_value(u.eV**-2.)
    print(delta, ' -> ',  phi)

print('Atmospheric neutrinos')
E = 1. * u.GeV
L = 1e3 * u.km
for delta in (DELTA_M21_CBE, DELTA_M32_CBE):
    phi = delta * (L / (4 * const.hbar * const.c * E)).to_value(u.eV**-2.)
    print(delta, ' -> ',  phi)


plt.show()
