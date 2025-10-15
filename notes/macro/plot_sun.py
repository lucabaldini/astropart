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

import numpy as np
import astropy.constants as const
import astropy.units as u
from astropy.units import cds

from astropart.plot import plt, standard_figure, margin_figure, \
    subplot_vertical_stack, setup_ca, save_cf
from astropart.ssm import BS05


# Quick estimate.
P_c = 32. / np.pi * const.G * const.M_sun**2 / const.R_sun**4.
print('%.2e' % P_c.to(cds.atm).value)
rho_c = 24. / np.pi * const.M_sun / const.R_sun**3.
print(rho_c.cgs)
T_c = 2. / 3. * const.G * const.M_sun / const.R_sun / const.R
print(T_c, '%.2e' % T_c.value)
P_rad = const.sigma_sb * (1.55e7 * u.K)**4 / 3.
print(const.sigma_sb)
#print(P_rad.to(u.J / u.m**3))


standard_figure('sun standard model', height=4.5)
axes = subplot_vertical_stack((1., 1., 1.))
plt.sca(axes[0])
BS05.cumulative_mass.plot(color='black')
setup_ca(ylabel='Cumulative mass [$M_\\odot$]')
plt.sca(axes[1])
BS05.density.plot(color='black')
setup_ca(ylabel='Density [g cm$^{-3}$]', logy=True)
plt.sca(axes[2])
BS05.temperature.plot(color='black')
setup_ca(xlabel='$r$ [$R\\odot$]', ylabel='Temperature [$^\\circ$K]', logy=True)
save_cf()

print(f'Temperature at the center of the Sun: {BS05.temperature(0.)} deg K')
print(f'Density at the center of the Sun: {BS05.density(0.)} g cm^{-3}')

margin_figure('sun_pm_fictional', height=2.5)
x = np.linspace(0., 1., 100)
y = 1. - x
plt.plot(x, y, color='k')
setup_ca(xlabel='$M~[M_\odot$]', ylabel='$P~[P_c]$', ymin=0)
save_cf()

plt.show()
